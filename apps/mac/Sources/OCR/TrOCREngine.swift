import AppKit
import CoreML

enum TrOCRError: Error {
    case resourceMissing(TrOCRModelSize)
    case preprocessingFailed
    case predictionFailed
}

/// Runs a TrOCR checkpoint (see tools/trocr/convert.py's DEFAULT_CHECKPOINTS
/// for which one per TrOCRModelSize) converted to CoreML
/// (Vendor/trocr/<size>/{encoder,decoder}.mlpackage, produced offline by
/// convert.py + export_tokenizer.py -- a one-time step per model, re-run
/// whenever a checkpoint changes). Greedy decode, one token per call, with
/// the decoder's self-attention KV cache held in a CoreML state (macOS 15)
/// -- see tools/trocr/kv_decoder.py for the mask contract.
///
/// small/base/large are English-only (official microsoft/trocr-*-printed).
/// khmer is a byte-BPE Khmer printed-text fine-tune (lkhapple/Khmer-TrOCR-OCR)
/// that is fed exactly the training preprocessing (see preprocess()). Tesseract remains available in Preferences
/// as an alternate Khmer option.
enum TrOCREngine {
    /// Points at the folder containing `<size>/{encoder,decoder}.mlpackage,tokenizer.json`.
    /// Normally the app's bundled `trocr` resource; tests set this to the
    /// repo's `Vendor/trocr` so inference can run in a hostless test bundle.
    static var resourceRootOverride: URL?

    private static var resourceRoot: URL? {
        resourceRootOverride
            ?? Bundle.main.resourceURL?.appendingPathComponent("trocr", isDirectory: true)
    }

    /// Total decoded ids incl. the start token; matches training
    /// MAX_LABEL_LEN / generation_config.max_length (128).
    static let maxLength = 128

    /// Greedy decode on this checkpoint can lock onto one token (measured: id
    /// 1437 "Ġ" repeated to max length on 2/20 printed fixtures) and sometimes
    /// recovers into real text afterwards. So instead of stopping, forbid a
    /// token from extending a run past maxRun: measured greedy CER 0.172 ->
    /// 0.056 (beam-4: 0.048). Labels never repeat one token more than 3x.
    /// ponytail: single-token runs only; 2-token cycles still end at maxLength.
    static let maxRun = 4

    /// Lowest Tesseract word confidence for it to replace a Latin-only TrOCR
    /// line. Measured: clean UI strips >= 75.9 (all correct); scanned eval
    /// lines Tesseract gets wrong <= 62.7 (one exception at 76.9, still
    /// lower CER than TrOCR there).
    static let tesseractMinConfidence = 70.0

    static func blockedToken(_ ids: [Int], maxRun: Int = maxRun) -> Int? {
        guard ids.count >= maxRun, let last = ids.last,
              ids.suffix(maxRun).allSatisfy({ $0 == last }) else { return nil }
        return last
    }

    /// Per-call timing breakdown for OCRBench diagnostics. ponytail: global,
    /// last-call-only -- fine for the single-threaded bench CLI; return it from
    /// recognizeText if the app ever needs it. With concurrent lines the ms
    /// totals add up overlapping time, so they can exceed the wall clock.
    struct RunStats {
        var strips = 0
        var decoderSteps = 0
        var encoderMs = 0.0
        var decoderMs = 0.0

        mutating func add(_ o: RunStats) {
            decoderSteps += o.decoderSteps
            encoderMs += o.encoderMs
            decoderMs += o.decoderMs
        }
    }
    static var lastRunStats = RunStats()

    /// Lines decoded at once. Bench knob: KHMERCLIPPER_TROCR_LINES.
    static let maxConcurrentLines = ProcessInfo.processInfo.environment["KHMERCLIPPER_TROCR_LINES"].flatMap(Int.init) ?? 4

    /// Running token total for the progress HUD across concurrent lines.
    @MainActor final class TokenCount {
        private var n = 0
        nonisolated init() {}
        func add(_ progress: @MainActor (OCRProgress) -> Void) {
            n += 1
            progress(.reading(tokens: n))
        }
    }

    static func milliseconds(since start: ContinuousClock.Instant) -> Double {
        let d = start.duration(to: .now)
        return Double(d.components.seconds) * 1_000 + Double(d.components.attoseconds) / 1e15
    }

    /// Warms the model cache ahead of an actual recognize call. Called when a
    /// capture starts so the slow first-time compile/load overlaps the user's
    /// selection drag instead of blocking after it. Best-effort: any error is
    /// swallowed and surfaced properly on the real recognizeText call.
    static func preload(model size: TrOCRModelSize) async {
        guard let root = resourceRoot else { return }
        _ = try? await ModelCache.shared.models(for: size, resourceRoot: root)
    }

    static func recognizeText(
        in image: CGImage,
        model size: TrOCRModelSize,
        progress: (@MainActor (OCRProgress) -> Void)? = nil
    ) async throws -> String {
        guard let root = resourceRoot else { throw TrOCRError.resourceMissing(size) }
        await progress?(.loadingModel)
        let (encoder, decoder, tokenizer) = try await ModelCache.shared.models(for: size, resourceRoot: root)

        await progress?(.encoding)
        // TrOCR reads one text line per pass, so a multi-line clip is split into
        // line strips and each decoded separately; joining with "\n" is what lets
        // Preserve Linebreaks have anything to preserve.
        let strips = lineStrips(image)
        // A long line squashed into 384x384 loses its thin marks; decode it in pieces.
        let pieces = strips.map { splitWide($0, maxAspect: wideSplitAspect) }
        let jobs = pieces.indices.flatMap { i in pieces[i].indices.map { (line: i, piece: $0) } }

        // Pieces decode concurrently (each has its own KV state): one decode
        // step is too small to fill the GPU, so overlapping lines is nearly free.
        let tokens = TokenCount()
        var texts = Array(repeating: [String](), count: strips.count)
        for i in strips.indices { texts[i] = Array(repeating: "", count: pieces[i].count) }
        var stats = RunStats(strips: strips.count)
        try await withThrowingTaskGroup(of: (Int, Int, String, RunStats).self) { group in
            var next = 0
            func addJob() {
                let (i, j) = jobs[next]
                let image = pieces[i][j].image
                group.addTask {
                    let (text, s) = try await recognizeLine(image, encoder: encoder, decoder: decoder, tokenizer: tokenizer) {
                        if let progress { await tokens.add(progress) }
                    }
                    return (i, j, text, s)
                }
                next += 1
            }
            while next < min(jobs.count, maxConcurrentLines) { addJob() }
            for try await (i, j, text, s) in group {
                texts[i][j] = text
                stats.add(s)
                if next < jobs.count { addJob() }
            }
        }
        lastRunStats = stats

        var lines: [String] = []
        for (i, strip) in strips.enumerated() {
            var line = ""
            for (j, piece) in pieces[i].enumerated() {
                let trimmed = texts[i][j].trimmingCharacters(in: .whitespaces)
                if !trimmed.isEmpty { line += (line.isEmpty ? "" : piece.separator) + trimmed }
            }
            // The Khmer fine-tune misspells clean English UI text ("Warrantly",
            // "PriceSnbox") that Tesseract reads perfectly, but Tesseract is worse
            // on low-res scanned English. So a Latin-only line switches to
            // Tesseract only when every word is confident. Lines with any Khmer
            // stay with TrOCR, which beats Tesseract on Khmer/mixed text.
            // ponytail: one tesseract process per Latin-only strip (~90ms each).
            var text = line
            if !containsKhmer(line),
               let tess = try? await TesseractEngine.recognizeLine(in: strip),
               !tess.text.isEmpty, tess.minConfidence >= tesseractMinConfidence {
                text = tess.text
            }
            if !text.isEmpty { lines.append(text) }
        }
        return lines.joined(separator: "\n")
    }

    private static func recognizeLine(
        _ image: CGImage,
        encoder: MLModel,
        decoder: MLModel,
        tokenizer: TrOCRTokenizer,
        onStep: @Sendable () async -> Void
    ) async throws -> (String, RunStats) {
        var stats = RunStats()
        let pixelValues = try preprocess(image: contentCrop(image))
        let encodeStart = ContinuousClock.now
        let encoderOutput = try await encoder.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "pixel_values": MLFeatureValue(multiArray: pixelValues),
        ]))
        stats.encoderMs = milliseconds(since: encodeStart)
        // The encoder model also projects the cross-attention K/V, once per line
        // instead of on every decoder step (tools/trocr/kv_decoder.py).
        guard let crossK = encoderOutput.featureValue(for: "cross_k")?.multiArrayValue,
              let crossV = encoderOutput.featureValue(for: "cross_v")?.multiArrayValue else {
            throw TrOCRError.predictionFailed
        }

        // Fresh KV state per line. Zero it: masked-out garbage could still be
        // NaN, and NaN * 0 survives both the mask and the one-hot blend.
        let state = decoder.makeState()
        for name in ["k_cache", "v_cache"] {
            state.withMultiArray(for: name) { cache in
                // Whole backing buffer incl. any stride padding -- one missed
                // element brings the NaN back.
                cache.withUnsafeMutableBytes { raw, _ in
                    if let base = raw.baseAddress { memset(base, 0, raw.count) }
                }
            }
        }

        // Mask contract (tools/trocr/kv_decoder.py): at step pos, attn_mask is
        // 0 for j <= pos and -1e4 after; write_mask is one-hot at pos.
        let n = NSNumber(value: maxLength)
        let inputIds = try MLMultiArray(shape: [1, 1], dataType: .int32)
        let position = try MLMultiArray(shape: [1], dataType: .int32)
        let attnMask = try MLMultiArray(shape: [1, 1, 1, n], dataType: .float32)
        let writeMask = try MLMultiArray(shape: [1, 1, n, 1], dataType: .float32)
        let attnPtr = attnMask.dataPointer.bindMemory(to: Float32.self, capacity: maxLength)
        let writePtr = writeMask.dataPointer.bindMemory(to: Float32.self, capacity: maxLength)
        for j in 0..<maxLength { attnPtr[j] = -1e4; writePtr[j] = 0 }
        let inputs = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: inputIds),
            "position": MLFeatureValue(multiArray: position),
            "attn_mask": MLFeatureValue(multiArray: attnMask),
            "write_mask": MLFeatureValue(multiArray: writeMask),
            "cross_k": MLFeatureValue(multiArray: crossK),
            "cross_v": MLFeatureValue(multiArray: crossV),
        ])

        let decodeStart = ContinuousClock.now
        var ids = [tokenizer.decoderStartId]
        for pos in 0..<(maxLength - 1) {
            try Task.checkCancellation()
            inputIds[0] = NSNumber(value: ids[pos])
            position[0] = NSNumber(value: pos)
            attnPtr[pos] = 0
            if pos > 0 { writePtr[pos - 1] = 0 }
            writePtr[pos] = 1

            let out = try await decoder.prediction(from: inputs, using: state)
            guard let logits = out.featureValue(for: "logits")?.multiArrayValue else {
                throw TrOCRError.predictionFailed
            }
            let nextId = argmaxLastStep(logits, excluding: blockedToken(ids))
            ids.append(nextId)
            await onStep()
            if nextId == tokenizer.eosId { break }
        }

        stats.decoderMs = milliseconds(since: decodeStart)
        stats.decoderSteps = ids.count - 1
        return (tokenizer.decode(ids: ids), stats)
    }

    /// Khmer (U+1780-17FF) or Khmer Symbols (U+19E0-19FF), the blocks the
    /// tokenizer adds per-character tokens for.
    static func containsKhmer(_ s: String) -> Bool {
        s.unicodeScalars.contains { (0x1780...0x17FF).contains($0.value) || (0x19E0...0x19FF).contains($0.value) }
    }

    static func argmaxLastStep(_ logits: MLMultiArray, excluding: Int? = nil) -> Int {
        let shape = logits.shape.map(\.intValue)
        let strides = logits.strides.map(\.intValue)
        let seqLen = shape[1]
        let vocabSize = shape[2]
        let base = (seqLen - 1) * strides[1]

        let ptr = logits.dataPointer.bindMemory(to: Float32.self, capacity: shape[0] * shape[1] * shape[2])
        var bestIndex = 0
        var bestValue = -Float.greatestFiniteMagnitude
        for v in 0..<vocabSize where v != excluding {
            let value = ptr[base + v * strides[2]]
            if value > bestValue {
                bestValue = value
                bestIndex = v
            }
        }
        return bestIndex
    }

    /// Downscaled (long edge <= scanMax) ink mask of the image. Background
    /// colour is taken from the image corners (works for both dark- and
    /// light-mode clips); any pixel differing from it by > 40/255 on any RGB
    /// channel counts as ink. `scale` maps mask coordinates back to full-res.
    private struct InkMask {
        let ink: [Bool]
        let width: Int, height: Int, scale: Int
        subscript(x: Int, y: Int) -> Bool { ink[y * width + x] }
    }

    private static func inkMask(_ image: CGImage, scanMax: Int = 512) -> InkMask? {
        let w = image.width, h = image.height
        guard w > 1, h > 1 else { return nil }
        let scale = max(1, (max(w, h) + scanMax - 1) / scanMax)
        let sw = w / scale, sh = h / scale
        guard sw > 1, sh > 1, let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }

        var px = [UInt8](repeating: 0, count: sw * sh * 4)
        guard let ctx = CGContext(data: &px, width: sw, height: sh, bitsPerComponent: 8,
                                  bytesPerRow: sw * 4, space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: sw, height: sh))

        func rgb(_ x: Int, _ y: Int) -> (Int, Int, Int) {
            let p = (y * sw + x) * 4
            return (Int(px[p]), Int(px[p + 1]), Int(px[p + 2]))
        }
        let corners = [rgb(0, 0), rgb(sw - 1, 0), rgb(0, sh - 1), rgb(sw - 1, sh - 1)]
        let bg = (corners.map(\.0).reduce(0, +) / 4,
                  corners.map(\.1).reduce(0, +) / 4,
                  corners.map(\.2).reduce(0, +) / 4)
        let threshold = 40

        var ink = [Bool](repeating: false, count: sw * sh)
        for y in 0..<sh {
            for x in 0..<sw {
                let (r, g, b) = rgb(x, y)
                ink[y * sw + x] = abs(r - bg.0) > threshold || abs(g - bg.1) > threshold || abs(b - bg.2) > threshold
            }
        }
        return InkMask(ink: ink, width: sw, height: sh, scale: scale)
    }

    /// Splits a clip into horizontal text-line strips by row ink projection:
    /// runs of inked rows separated by blank rows. At normal Khmer line
    /// heights the subscripts of one line touch the above-marks of the next,
    /// so a run can hold several lines; those runs are first cut at the line
    /// pitch (`touchingCuts`). Runs much shorter than the tallest (a detached
    /// Khmer vowel/subscript mark, a stray dot) are then merged into the
    /// nearer neighbour rather than read as their own line -- splitting first
    /// matters, or a merged double line makes real lines look like marks.
    /// Strips extend halfway into the surrounding gaps so marks above/below
    /// the line stay attached. Returns [image] when there is 0 or 1 line.
    /// Mirrors training/eval/src/khmerclipper_eval/pipeline.py line_strips
    /// (split_touching=True), which measured multi-line screen-clip CER
    /// 0.314 -> 0.046 (training/eval/results/accuracy_v3.md).
    static func lineStrips(_ image: CGImage) -> [CGImage] {
        // Full resolution: line pitch on small screen text is only ~20 rows.
        guard let mask = inkMask(image, scanMax: max(image.width, image.height)) else { return [image] }
        var counts = [Double](repeating: 0, count: mask.height)
        var runs: [(lo: Int, hi: Int)] = []
        for y in 0..<mask.height {
            let n = (0..<mask.width).reduce(0) { $0 + (mask[$1, y] ? 1 : 0) }
            counts[y] = Double(n)
            if n == 0 { continue }
            if let last = runs.last, last.hi == y - 1 {
                runs[runs.count - 1].hi = y
            } else {
                runs.append((y, y))
            }
        }
        // 3-row box smoothing, zero-padded at the ends (numpy convolve "same").
        let profile = counts.indices.map { y in
            ((y > 0 ? counts[y - 1] : 0) + counts[y] + (y + 1 < counts.count ? counts[y + 1] : 0)) / 3
        }
        if !runs.isEmpty {
            let pitch = linePitch(profile)
            runs = runs.flatMap { run -> [(lo: Int, hi: Int)] in
                let seg = Array(profile[run.lo...run.hi])
                let cuts: [Int]
                if let pitch {
                    cuts = Double(seg.count) >= 1.5 * Double(pitch) ? touchingCuts(seg, pitch: pitch) : []
                } else {
                    cuts = twoLineCut(seg).map { [$0] } ?? []
                }
                let edges = [run.lo] + cuts.map { run.lo + $0 } + [run.hi + 1]
                return (0..<(edges.count - 1)).map { (edges[$0], edges[$0 + 1] - 1) }
            }
        }
        runs = mergeMarks(runs)
        guard runs.count > 1 else { return [image] }

        return runs.indices.compactMap { k in
            let top = k == 0 ? 0 : (runs[k - 1].hi + runs[k].lo + 1) / 2
            let bottom = k == runs.count - 1 ? mask.height : (runs[k].hi + runs[k + 1].lo + 1) / 2
            let rect = CGRect(x: 0, y: top * mask.scale, width: image.width,
                              height: min(image.height, bottom * mask.scale) - top * mask.scale)
            return image.cropping(to: rect)
        }
    }

    /// Autocorrelation peak a line pitch needs, and how sparse (vs the run's
    /// densest row) a cut between touching lines must be.
    static let pitchStrength = 0.1
    static let touchingValley = 0.2

    /// Line pitch in rows: the first autocorrelation peak of the row-ink
    /// profile above `pitchStrength`, searched between n/12 and n/1.5 rows.
    /// nil when the clip shows no repeating line structure (one line).
    static func linePitch(_ profile: [Double]) -> Int? {
        let n = profile.count
        guard n >= 8 else { return nil }
        let mean = profile.reduce(0, +) / Double(n)
        let x = profile.map { $0 - mean }
        func ac(_ k: Int) -> Double { (0..<(n - k)).reduce(0) { $0 + x[$1 + k] * x[$1] } }
        let ac0 = ac(0)
        guard ac0 > 0 else { return nil }
        let lo = max(4, n / 12), hi = min(n - 1, Int(Double(n) / 1.5) + 1)
        guard lo < hi else { return nil }
        var prev = ac(lo - 1) / ac0, cur = ac(lo) / ac0
        for k in lo..<hi {
            let next = ac(k + 1) / ac0
            if cur >= prev && cur >= next && cur > pitchStrength { return k }
            (prev, cur) = (cur, next)
        }
        return nil
    }

    /// Rows (relative to the run) at which to cut a run holding several
    /// touching lines: the sparsest row near each expected boundary, kept
    /// only when it is <= `touchingValley` x the run's densest row.
    static func touchingCuts(_ profile: [Double], pitch: Int) -> [Int] {
        let h = profile.count, p = Double(pitch)
        let n = Int(((Double(h) - 0.5 * p) / p).rounded(.down)) + 1
        guard n >= 2, let peak = profile.max() else { return [] }
        let lineH = Double(h - (n - 1) * pitch)
        var cuts: [Int] = []
        for i in 1..<n {
            let centre = Double(i - 1) * p + (p + lineH) / 2
            let a = Int(max(1, centre - p / 3)), b = Int(min(Double(h - 1), centre + p / 3 + 1))
            guard a < b else { continue }
            var c = a
            for y in a..<b where profile[y] < profile[c] { c = y }
            if profile[c] <= touchingValley * peak { cuts.append(c) }
        }
        return cuts
    }

    /// A mid-run cut with no pitch must be this sparse vs the lighter line, and
    /// the lighter line's densest row this dense vs the run's.
    static let twoLineValley = 0.15
    static let twoLineBand = 0.3

    /// Row to cut a run holding two touching lines when linePitch finds no
    /// pitch (two lines, the second often short, are too little repetition
    /// for the autocorrelation): the sparsest row in the middle 35-65%, kept
    /// only when both sides carry a real line of ink. Screen set: 439 -> 445
    /// of 450 clips segmented right, no new splits on the frozen 200 or slices.
    static func twoLineCut(_ profile: [Double]) -> Int? {
        let h = profile.count
        let a = max(1, Int(Double(h) * 0.35)), b = min(h - 1, Int(Double(h) * 0.65) + 1)
        guard h >= 9, b - a >= 2, let peak = profile.max() else { return nil }
        var c = a
        for y in a..<b where profile[y] < profile[c] { c = y }
        let weak = min(profile[..<c].max() ?? 0, profile[(c + 1)...].max() ?? 0)
        guard weak >= twoLineBand * peak, profile[c] <= twoLineValley * weak else { return nil }
        return c
    }

    /// ponytail: fixed cutoffs. Merge a run < 35% of the tallest, or < 50%
    /// when it sits within 25% of the tallest's height of a neighbour --
    /// Khmer above-base marks form runs up to ~46% of the base line 1-3px
    /// above it (measured: the old 35%-only rule split 17/200 single-line
    /// eval fixtures), while real line gaps are far larger. Tune if
    /// headings next to small body text get merged.
    static func mergeMarks(_ input: [(lo: Int, hi: Int)]) -> [(lo: Int, hi: Int)] {
        var runs = input
        while runs.count > 1 {
            let tallest = runs.map { $0.hi - $0.lo + 1 }.max()!
            func nearestGap(_ k: Int) -> Int {
                min(k > 0 ? runs[k].lo - runs[k - 1].hi : Int.max,
                    k < runs.count - 1 ? runs[k + 1].lo - runs[k].hi : Int.max)
            }
            guard let i = runs.indices.first(where: { k in
                let h = (runs[k].hi - runs[k].lo + 1) * 100
                return h < tallest * 35 || (h < tallest * 50 && nearestGap(k) < Int.max && nearestGap(k) * 4 < tallest)
            }) else { break }
            let gapAbove = i > 0 ? runs[i].lo - runs[i - 1].hi : Int.max
            let gapBelow = i < runs.count - 1 ? runs[i + 1].lo - runs[i].hi : Int.max
            let j = gapAbove <= gapBelow ? i - 1 : i + 1
            runs[min(i, j)] = (min(runs[i].lo, runs[j].lo), max(runs[i].hi, runs[j].hi))
            runs.remove(at: max(i, j))
        }
        return runs
    }

    /// Trims the background margin around the ink (plus 4% padding) so the
    /// text fills the 384x384 input. Strips span the whole clip width, and a
    /// short line's blank tail otherwise wastes most of the input; screen-set
    /// errors -22% (training/eval/results/accuracy_v3.md). A mis-detected
    /// background collapses the box to a sliver (< 1/20 of a side): keep the
    /// image then. Port of apps/desktop preprocess.content_crop.
    static func contentCrop(_ image: CGImage) -> CGImage {
        guard let mask = inkMask(image) else { return image }
        var minX = Int.max, maxX = -1, minY = Int.max, maxY = -1
        for y in 0..<mask.height {
            for x in 0..<mask.width where mask[x, y] {
                minX = min(minX, x); maxX = max(maxX, x); minY = min(minY, y); maxY = max(maxY, y)
            }
        }
        guard maxX >= 0, (maxX - minX + 1) * 20 >= mask.width, (maxY - minY + 1) * 20 >= mask.height else {
            return image
        }
        let padX = max(1, (maxX - minX + 1) * 4 / 100), padY = max(1, (maxY - minY + 1) * 4 / 100)
        let s = mask.scale
        let x0 = max(0, (minX - padX) * s), y0 = max(0, (minY - padY) * s)
        let x1 = min(image.width, (maxX + 1 + padX) * s), y1 = min(image.height, (maxY + 1 + padY) * s)
        return image.cropping(to: CGRect(x: x0, y: y0, width: x1 - x0, height: y1 - y0)) ?? image
    }

    /// Lines wider than this many strip heights are decoded in pieces.
    /// Swept 8 / 12 / 16 (training/eval/results/accuracy_v3.md): 12 cut
    /// screen-set errors 20% without hurting single lines; 8 split 8-12:1
    /// lines that read fine whole.
    static let wideSplitAspect = 12.0

    /// Cuts a strip whose *line* is wider than `maxAspect` x the strip height
    /// into ~equal pieces at blank-column runs. The line's extent comes from
    /// ink in the strip's central band (rows 30-70%): a strip spans the whole
    /// clip width and its edges hold the neighbour lines' marks, and a piece
    /// of that blank tail makes TrOCR hallucinate. Each ideal cut snaps to the
    /// nearest blank run within a quarter piece width, or is skipped (never
    /// cuts ink). The separator is " " for a gap >= a quarter of the text
    /// height, else "" (Khmer words often have no space).
    /// Mirrors training/eval/src/khmerclipper_eval/pipeline.py split_wide.
    static func splitWide(_ strip: CGImage, maxAspect: Double) -> [(separator: String, image: CGImage)] {
        let w = strip.width, h = strip.height
        let whole = [(separator: "", image: strip)]
        guard Double(w) > maxAspect * Double(h),
              let mask = inkMask(strip, scanMax: max(w, h)), mask.width == w, mask.height == h else { return whole }
        let bandTop = Int(0.3 * Double(h)), bandBottom = max(bandTop + 1, Int(0.7 * Double(h)))
        let coreCols = (0..<w).filter { x in (bandTop..<bandBottom).contains { mask[x, $0] } }
        guard let first = coreCols.first, let last = coreCols.last else { return whole }
        let pad = max(2, h / 5)
        let x0 = max(0, first - pad), x1 = min(w, last + 1 + pad)
        guard Double(x1 - x0) > maxAspect * Double(h) else { return whole }
        let inkRows = (0..<h).filter { y in (0..<w).contains { mask[$0, y] } }
        let textH = (inkRows.last ?? h - 1) - (inkRows.first ?? 0) + 1
        let blank = (0..<w).map { x in !(0..<h).contains { mask[x, $0] } }
        var gaps: [(Int, Int)] = []   // [start, end) blank-column runs inside the line
        var x = x0
        while x < x1 {
            if blank[x] {
                let start = x
                while x < x1 && blank[x] { x += 1 }
                if start > x0 && x < x1 { gaps.append((start, x)) }
            } else {
                x += 1
            }
        }
        let n = Int((Double(x1 - x0) / (maxAspect * Double(h))).rounded(.up))
        let pieceW = Double(x1 - x0) / Double(n)
        var cuts: [(x: Int, separator: String)] = []
        for k in 1..<max(1, n) {
            let ideal = Double(x0) + Double(k) * pieceW
            let centre = { (g: (Int, Int)) in Double(g.0 + g.1) / 2 }
            let near = gaps.filter { abs(centre($0) - ideal) <= 0.25 * pieceW && (cuts.isEmpty || $0.0 > cuts.last!.x) }
            guard let g = near.min(by: { abs(centre($0) - ideal) < abs(centre($1) - ideal) }) else { continue }
            cuts.append(((g.0 + g.1) / 2, Double(g.1 - g.0) >= 0.25 * Double(textH) ? " " : ""))
        }
        guard !cuts.isEmpty else { return whole }
        var pieces: [(separator: String, image: CGImage)] = []
        var left = x0, separator = ""
        for cut in cuts + [(x: x1, separator: "")] {
            if let piece = strip.cropping(to: CGRect(x: left, y: 0, width: cut.x - left, height: h)) {
                pieces.append((separator, piece))
            }
            left = cut.x
            separator = cut.separator
        }
        return pieces
    }

    /// RGB bytes resized exactly like PIL's `Image.resize(size, BILINEAR)` --
    /// the call training uses (trocr_khmer.dataset.OCRDataset). Separable
    /// triangle filter whose support widens by the scale factor when
    /// downsampling (antialias), horizontal pass then vertical, each rounded
    /// to UInt8 like PIL's 8-bit path. Mirrors Pillow's Resample.c
    /// precompute_coeffs.
    static func resizedRGB(_ image: CGImage, size: Int) -> [UInt8]? {
        let w = image.width, h = image.height
        // Draw in the image's own RGB space so values pass through unconverted,
        // like PIL (which ignores colour profiles); sRGB only for non-RGB images.
        guard w > 0, h > 0, let colorSpace = image.colorSpace.flatMap({ $0.model == .rgb ? $0 : nil })
                ?? CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        var src = [UInt8](repeating: 0, count: w * h * 4)
        guard let ctx = CGContext(data: &src, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: colorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))

        // (start index, normalized weights) per output pixel along one axis.
        func coeffs(inSize: Int, outSize: Int) -> [(Int, [Double])] {
            let scale = Double(inSize) / Double(outSize)
            let filterScale = max(scale, 1.0)
            let support = filterScale   // bilinear support 1.0 * filterScale
            return (0..<outSize).map { i in
                let center = (Double(i) + 0.5) * scale
                let lo = max(Int(center - support + 0.5), 0)
                let hi = min(Int(center + support + 0.5), inSize)
                var ws = (lo..<hi).map { x in max(0, 1 - abs((Double(x) - center + 0.5) / filterScale)) }
                let total = ws.reduce(0, +)
                if total > 0 { ws = ws.map { $0 / total } }
                return (lo, ws)
            }
        }
        func clip8(_ v: Double) -> UInt8 { UInt8(max(0, min(255, v.rounded()))) }

        let cx = coeffs(inSize: w, outSize: size), cy = coeffs(inSize: h, outSize: size)
        var tmp = [UInt8](repeating: 0, count: size * h * 3)   // horizontal pass
        for y in 0..<h {
            for (ox, (lo, ws)) in cx.enumerated() {
                for c in 0..<3 {
                    var acc = 0.0
                    for (k, wt) in ws.enumerated() { acc += wt * Double(src[(y * w + lo + k) * 4 + c]) }
                    tmp[(y * size + ox) * 3 + c] = clip8(acc)
                }
            }
        }
        var out = [UInt8](repeating: 0, count: size * size * 3)  // vertical pass
        for (oy, (lo, ws)) in cy.enumerated() {
            for x in 0..<size {
                for c in 0..<3 {
                    var acc = 0.0
                    for (k, wt) in ws.enumerated() { acc += wt * Double(tmp[((lo + k) * size + x) * 3 + c]) }
                    out[(oy * size + x) * 3 + c] = clip8(acc)
                }
            }
        }
        return out
    }

    /// ViTImageProcessor preprocessing, identical to training: PIL-bilinear
    /// resize to 384x384 (no aspect crop, no content crop -- training images are
    /// never cropped, and cropping measured +0.06 CER), then normalize with
    /// mean=std=0.5.
    private static func preprocess(image: CGImage, size: Int = 384) throws -> MLMultiArray {
        guard let rgb = resizedRGB(image, size: size) else { throw TrOCRError.preprocessingFailed }
        let array = try MLMultiArray(shape: [1, 3, NSNumber(value: size), NSNumber(value: size)], dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float32.self, capacity: size * size * 3)
        let plane = size * size
        for idx in 0..<plane {
            for c in 0..<3 {
                ptr[c * plane + idx] = Float32(rgb[idx * 3 + c]) / 127.5 - 1.0
            }
        }
        return array
    }

    /// Compiles + caches the CoreML models and tokenizer per size. `.mlpackage`
    /// is bundled raw (a plain resource copy, not Xcode-compiled) so it's
    /// compiled to `.mlmodelc` on first use and the result persisted under
    /// Application Support to avoid recompiling on every launch.
    private actor ModelCache {
        static let shared = ModelCache()

        /// Bench knob: KHMERCLIPPER_TROCR_UNITS=all|gpu|ane|cpu. ponytail: env
        /// read, not a preference. Default `.cpuAndGPU` (2026-10-03 full-200:
        /// CER 0.0593 tied with `.all`, median 95.3 ms vs 105.9 ms).
        private static let computeUnits: MLComputeUnits = {
            switch ProcessInfo.processInfo.environment["KHMERCLIPPER_TROCR_UNITS"] {
            case "all": return .all
            case "ane": return .cpuAndNeuralEngine
            case "cpu": return .cpuOnly
            default: return .cpuAndGPU
            }
        }()

        private var loaded: [TrOCRModelSize: (encoder: MLModel, decoder: MLModel, tokenizer: TrOCRTokenizer)] = [:]

        func models(for size: TrOCRModelSize, resourceRoot: URL) throws -> (MLModel, MLModel, TrOCRTokenizer) {
            if let cached = loaded[size] { return (cached.encoder, cached.decoder, cached.tokenizer) }

            let dir = resourceRoot.appendingPathComponent(size.rawValue, isDirectory: true)
            // The tag names the model I/O contract: a cache compiled from an older
            // package (encoder_hidden_states input) must never be matched, even if
            // its mtime is newer than the package (DMG copies keep mtimes).
            let tag = "\(size.rawValue)-xkv"
            let encoder = try Self.loadCompiled(packageURL: dir.appendingPathComponent("encoder.mlpackage"), name: "\(tag)-encoder")
            let decoder = try Self.loadCompiled(packageURL: dir.appendingPathComponent("decoder.mlpackage"), name: "\(tag)-decoder")
            let tokenizerData = try Data(contentsOf: dir.appendingPathComponent("tokenizer.json"))
            let tokenizer = try TrOCRTokenizer(data: tokenizerData)

            loaded[size] = (encoder, decoder, tokenizer)
            return (encoder, decoder, tokenizer)
        }

        private static func loadCompiled(packageURL: URL, name: String) throws -> MLModel {
            let cacheDir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("KhmerClipper/TrOCRCompiled", isDirectory: true)
            try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)

            let compiledURL = cacheDir.appendingPathComponent("\(name).mlmodelc")
            // Recompile when the cache is missing OR older than the source
            // package: an app update (or a re-run of convert.py) that swaps in a
            // new .mlpackage for the same size must not keep serving the stale
            // compiled model -- that silently produced wrong output when the
            // khmer checkpoint changed.
            let fm = FileManager.default
            if !fm.fileExists(atPath: compiledURL.path) || Self.isOlder(compiledURL, than: packageURL) {
                let freshlyCompiled = try MLModel.compileModel(at: packageURL)
                if fm.fileExists(atPath: compiledURL.path) {
                    try fm.removeItem(at: compiledURL)
                }
                try fm.moveItem(at: freshlyCompiled, to: compiledURL)
            }
            let config = MLModelConfiguration()
            config.computeUnits = Self.computeUnits
            return try MLModel(contentsOf: compiledURL, configuration: config)
        }

        private static func isOlder(_ url: URL, than other: URL) -> Bool {
            let fm = FileManager.default
            guard let a = try? fm.attributesOfItem(atPath: url.path)[.modificationDate] as? Date,
                  let b = try? fm.attributesOfItem(atPath: other.path)[.modificationDate] as? Date else {
                return true   // can't tell -> recompile, the safe choice
            }
            return a < b
        }
    }
}
