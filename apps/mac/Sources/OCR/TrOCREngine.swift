import AppKit
import CoreML

enum TrOCRError: Error {
    case resourceMissing(TrOCRModelSize)
    case preprocessingFailed
    case predictionFailed
}

/// Runs a TrOCR checkpoint (see Vendor/trocr/convert.py's DEFAULT_CHECKPOINTS
/// for which one per TrOCRModelSize) converted to CoreML
/// (Vendor/trocr/<size>/{encoder,decoder}.mlpackage, produced offline by
/// convert.py + export_tokenizer.py -- a one-time step per model, re-run
/// whenever a checkpoint changes). Greedy, no-KV-cache decode loop: OCR text
/// regions are short enough that re-running the full decoder each step is
/// simple and fast enough, without CoreML's more complex stateful-model
/// support.
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

    static func blockedToken(_ ids: [Int], maxRun: Int = maxRun) -> Int? {
        guard ids.count >= maxRun, let last = ids.last,
              ids.suffix(maxRun).allSatisfy({ $0 == last }) else { return nil }
        return last
    }

    /// Per-call timing breakdown for OCRBench diagnostics. ponytail: global,
    /// last-call-only -- fine for the single-threaded bench CLI; return it from
    /// recognizeText if the app ever needs it.
    struct RunStats {
        var strips = 0
        var decoderSteps = 0
        var encoderMs = 0.0
        var decoderMs = 0.0
    }
    static var lastRunStats = RunStats()

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
        var lines: [String] = []
        var doneTokens = 0
        lastRunStats = RunStats()
        let strips = lineStrips(image)
        lastRunStats.strips = strips.count
        for strip in strips {
            var lineTokens = 0
            let line = try await recognizeLine(strip, encoder: encoder, decoder: decoder, tokenizer: tokenizer) { n in
                lineTokens = n
                await progress?(.reading(tokens: doneTokens + n))
            }
            doneTokens += lineTokens
            if !line.isEmpty { lines.append(line) }
        }
        return lines.joined(separator: "\n")
    }

    private static func recognizeLine(
        _ image: CGImage,
        encoder: MLModel,
        decoder: MLModel,
        tokenizer: TrOCRTokenizer,
        onStep: (Int) async -> Void
    ) async throws -> String {
        let pixelValues = try preprocess(image: image)
        let encodeStart = ContinuousClock.now
        let encoderOutput = try await encoder.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "pixel_values": MLFeatureValue(multiArray: pixelValues),
        ]))
        lastRunStats.encoderMs += milliseconds(since: encodeStart)
        guard let encoderHidden = encoderOutput.featureValue(for: "encoder_hidden_states")?.multiArrayValue else {
            throw TrOCRError.predictionFailed
        }

        let decodeStart = ContinuousClock.now
        var ids = [tokenizer.decoderStartId]
        while ids.count < maxLength {
            try Task.checkCancellation()
            let inputIds = try MLMultiArray(shape: [1, NSNumber(value: ids.count)], dataType: .int32)
            let idsPtr = inputIds.dataPointer.bindMemory(to: Int32.self, capacity: ids.count)
            for (i, id) in ids.enumerated() { idsPtr[i] = Int32(id) }

            let decoderOutput = try await decoder.prediction(from: MLDictionaryFeatureProvider(dictionary: [
                "input_ids": MLFeatureValue(multiArray: inputIds),
                "encoder_hidden_states": MLFeatureValue(multiArray: encoderHidden),
            ]))
            guard let logits = decoderOutput.featureValue(for: "logits")?.multiArrayValue else {
                throw TrOCRError.predictionFailed
            }
            let nextId = argmaxLastStep(logits, excluding: blockedToken(ids))
            ids.append(nextId)
            await onStep(ids.count)
            if nextId == tokenizer.eosId { break }
        }

        lastRunStats.decoderMs += milliseconds(since: decodeStart)
        lastRunStats.decoderSteps += ids.count - 1
        return tokenizer.decode(ids: ids)
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
    /// runs of inked rows separated by blank rows. Runs much shorter than the
    /// tallest (a detached Khmer vowel/subscript mark, a stray dot) are merged
    /// into the nearer neighbour rather than read as their own line. Strips
    /// extend halfway into the surrounding gaps so marks above/below the line
    /// stay attached. Returns [image] when there is 0 or 1 line.
    static func lineStrips(_ image: CGImage) -> [CGImage] {
        guard let mask = inkMask(image) else { return [image] }
        var runs: [(lo: Int, hi: Int)] = []
        for y in 0..<mask.height {
            let inked = (0..<mask.width).contains { mask[$0, y] }
            if !inked { continue }
            if let last = runs.last, last.hi == y - 1 {
                runs[runs.count - 1].hi = y
            } else {
                runs.append((y, y))
            }
        }
        // ponytail: fixed cutoffs. Merge a run < 35% of the tallest, or < 50%
        // when it sits within 25% of the tallest's height of a neighbour --
        // Khmer above-base marks form runs up to ~46% of the base line 1-3px
        // above it (measured: the old 35%-only rule split 17/200 single-line
        // eval fixtures), while real line gaps are far larger. Tune if
        // headings next to small body text get merged.
        while runs.count > 1 {
            let tallest = runs.map { $0.hi - $0.lo + 1 }.max()!
            func nearestGap(_ k: Int) -> Int {
                min(k > 0 ? runs[k].lo - runs[k - 1].hi : Int.max,
                    k < runs.count - 1 ? runs[k + 1].lo - runs[k].hi : Int.max)
            }
            guard let i = runs.indices.first(where: { k in
                let h = (runs[k].hi - runs[k].lo + 1) * 100
                return h < tallest * 35 || (h < tallest * 50 && nearestGap(k) * 4 < tallest)
            }) else { break }
            let gapAbove = i > 0 ? runs[i].lo - runs[i - 1].hi : Int.max
            let gapBelow = i < runs.count - 1 ? runs[i + 1].lo - runs[i].hi : Int.max
            let j = gapAbove <= gapBelow ? i - 1 : i + 1
            runs[min(i, j)] = (min(runs[i].lo, runs[j].lo), max(runs[i].hi, runs[j].hi))
            runs.remove(at: max(i, j))
        }
        guard runs.count > 1 else { return [image] }

        return runs.indices.compactMap { k in
            let top = k == 0 ? 0 : (runs[k - 1].hi + runs[k].lo + 1) / 2
            let bottom = k == runs.count - 1 ? mask.height : (runs[k].hi + runs[k + 1].lo + 1) / 2
            let rect = CGRect(x: 0, y: top * mask.scale, width: image.width,
                              height: min(image.height, bottom * mask.scale) - top * mask.scale)
            return image.cropping(to: rect)
        }
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

        private var loaded: [TrOCRModelSize: (encoder: MLModel, decoder: MLModel, tokenizer: TrOCRTokenizer)] = [:]

        func models(for size: TrOCRModelSize, resourceRoot: URL) throws -> (MLModel, MLModel, TrOCRTokenizer) {
            if let cached = loaded[size] { return (cached.encoder, cached.decoder, cached.tokenizer) }

            let dir = resourceRoot.appendingPathComponent(size.rawValue, isDirectory: true)
            let encoder = try Self.loadCompiled(packageURL: dir.appendingPathComponent("encoder.mlpackage"), name: "\(size.rawValue)-encoder")
            let decoder = try Self.loadCompiled(packageURL: dir.appendingPathComponent("decoder.mlpackage"), name: "\(size.rawValue)-decoder")
            let tokenizerData = try Data(contentsOf: dir.appendingPathComponent("tokenizer.json"))
            let tokenizer = try TrOCRTokenizer(data: tokenizerData)

            loaded[size] = (encoder, decoder, tokenizer)
            return (encoder, decoder, tokenizer)
        }

        private static func loadCompiled(packageURL: URL, name: String) throws -> MLModel {
            let cacheDir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("KhmerClip/TrOCRCompiled", isDirectory: true)
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
            return try MLModel(contentsOf: compiledURL)
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
