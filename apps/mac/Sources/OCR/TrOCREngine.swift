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
/// that reads printed Khmer when the text fills the frame -- preprocess()
/// content-crops to make that so. Tesseract remains available in Preferences
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

    private static let maxLength = 64

    static func recognizeText(
        in image: CGImage,
        model size: TrOCRModelSize,
        progress: (@MainActor (OCRProgress) -> Void)? = nil
    ) async throws -> String {
        guard let root = resourceRoot else { throw TrOCRError.resourceMissing(size) }
        await progress?(.loadingModel)
        let (encoder, decoder, tokenizer) = try await ModelCache.shared.models(for: size, resourceRoot: root)

        await progress?(.encoding)
        let pixelValues = try preprocess(image: image)
        let encoderOutput = try await encoder.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "pixel_values": MLFeatureValue(multiArray: pixelValues),
        ]))
        guard let encoderHidden = encoderOutput.featureValue(for: "encoder_hidden_states")?.multiArrayValue else {
            throw TrOCRError.predictionFailed
        }

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
            let nextId = argmaxLastStep(logits)
            ids.append(nextId)
            await progress?(.reading(Double(ids.count) / Double(maxLength)))
            if nextId == tokenizer.eosId { break }
        }

        return tokenizer.decode(ids: ids)
    }

    private static func argmaxLastStep(_ logits: MLMultiArray) -> Int {
        let shape = logits.shape.map(\.intValue)
        let strides = logits.strides.map(\.intValue)
        let seqLen = shape[1]
        let vocabSize = shape[2]
        let base = (seqLen - 1) * strides[1]

        let ptr = logits.dataPointer.bindMemory(to: Float32.self, capacity: shape[0] * shape[1] * shape[2])
        var bestIndex = 0
        var bestValue = -Float.greatestFiniteMagnitude
        for v in 0..<vocabSize {
            let value = ptr[base + v * strides[2]]
            if value > bestValue {
                bestValue = value
                bestIndex = v
            }
        }
        return bestIndex
    }

    /// Trims uniform background margin around the content. Background colour is
    /// taken from the image corners (works for both dark- and light-mode clips),
    /// and any pixel differing from it beyond a threshold counts as content. The
    /// content bounding box, padded slightly, is cropped out; if nothing stands
    /// out from the background the original image is returned unchanged. Scans a
    /// downscaled copy (long edge <= scanMax) so cost stays bounded on big clips.
    private static func contentCrop(_ image: CGImage, scanMax: Int = 512) -> CGImage {
        let w = image.width, h = image.height
        guard w > 1, h > 1 else { return image }
        let scale = max(1, (max(w, h) + scanMax - 1) / scanMax)
        let sw = w / scale, sh = h / scale
        guard sw > 1, sh > 1, let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return image }

        var px = [UInt8](repeating: 0, count: sw * sh * 4)
        guard let ctx = CGContext(data: &px, width: sw, height: sh, bitsPerComponent: 8,
                                  bytesPerRow: sw * 4, space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return image }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: sw, height: sh))

        // Background = average of the four corners. Content = differs by > 40/255
        // on any RGB channel.
        func rgb(_ x: Int, _ y: Int) -> (Int, Int, Int) {
            let p = (y * sw + x) * 4
            return (Int(px[p]), Int(px[p + 1]), Int(px[p + 2]))
        }
        let corners = [rgb(0, 0), rgb(sw - 1, 0), rgb(0, sh - 1), rgb(sw - 1, sh - 1)]
        let bg = (corners.map(\.0).reduce(0, +) / 4,
                  corners.map(\.1).reduce(0, +) / 4,
                  corners.map(\.2).reduce(0, +) / 4)
        let threshold = 40

        var minX = sw, minY = sh, maxX = -1, maxY = -1
        for y in 0..<sh {
            for x in 0..<sw {
                let (r, g, b) = rgb(x, y)
                if abs(r - bg.0) > threshold || abs(g - bg.1) > threshold || abs(b - bg.2) > threshold {
                    if x < minX { minX = x }
                    if x > maxX { maxX = x }
                    if y < minY { minY = y }
                    if y > maxY { maxY = y }
                }
            }
        }
        guard maxX >= minX, maxY >= minY else { return image }
        // Guard against a mis-detected background (e.g. a corner sitting on ink)
        // collapsing the box to a sliver: if content spans < 5% of either axis,
        // don't trust the crop -- fall back to the full image.
        if (maxX - minX + 1) * 20 < sw || (maxY - minY + 1) * 20 < sh { return image }

        // Map the scan-space bbox back to full-res, pad by 4% of the box size.
        let padX = max(1, (maxX - minX + 1) * 4 / 100)
        let padY = max(1, (maxY - minY + 1) * 4 / 100)
        let x0 = max(0, (minX - padX) * scale)
        let y0 = max(0, (minY - padY) * scale)
        let x1 = min(w, (maxX + 1 + padX) * scale)
        let y1 = min(h, (maxY + 1 + padY) * scale)
        let rect = CGRect(x: x0, y: y0, width: x1 - x0, height: y1 - y0)
        return image.cropping(to: rect) ?? image
    }

    /// Mirrors the HF `ViTImageProcessor`/`DeiTImageProcessor` preprocessing
    /// used by all three `-printed` checkpoints: resize to 384x384 (no aspect
    /// crop), normalize with mean=std=0.5 per channel. First trims uniform
    /// background margin (see contentCrop) so the text fills the frame -- TrOCR
    /// is trained on tightly-cropped text lines and, given a loose crop with
    /// lots of background, keeps decoding past the real text into garbage.
    private static func preprocess(image rawImage: CGImage, size: Int = 384) throws -> MLMultiArray {
        let image = contentCrop(rawImage)
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { throw TrOCRError.preprocessingFailed }
        var pixels = [UInt8](repeating: 0, count: size * size * 4)
        guard let context = CGContext(
            data: &pixels,
            width: size,
            height: size,
            bitsPerComponent: 8,
            bytesPerRow: size * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw TrOCRError.preprocessingFailed
        }
        // .medium (roughly bilinear) was measured closest to the HF processor's
        // resize among CGContext's quality levels; CGContext's resampler is a
        // different implementation than PIL's, so this narrows but doesn't
        // eliminate the pixel-value gap between on-device and reference output.
        context.interpolationQuality = .medium
        context.draw(image, in: CGRect(x: 0, y: 0, width: size, height: size))

        let array = try MLMultiArray(shape: [1, 3, NSNumber(value: size), NSNumber(value: size)], dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float32.self, capacity: size * size * 3)
        let plane = size * size
        for y in 0..<size {
            for x in 0..<size {
                let p = (y * size + x) * 4
                let idx = y * size + x
                ptr[idx] = (Float32(pixels[p]) / 255.0 - 0.5) / 0.5
                ptr[plane + idx] = (Float32(pixels[p + 1]) / 255.0 - 0.5) / 0.5
                ptr[2 * plane + idx] = (Float32(pixels[p + 2]) / 255.0 - 0.5) / 0.5
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
                .appendingPathComponent("Netra/TrOCRCompiled", isDirectory: true)
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
