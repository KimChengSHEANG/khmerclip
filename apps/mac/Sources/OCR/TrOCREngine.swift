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
/// khmer is a Khmer fine-tune (KiteAether/khmer-trcor-06-10) with a reported
/// CER of 1.13 on its model card as of this writing -- an early/WIP result,
/// not a reliable OCR model yet. Tesseract remains available in Preferences
/// as the dependable Khmer option.
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

    static func recognizeText(in image: CGImage, model size: TrOCRModelSize) async throws -> String {
        guard let root = resourceRoot else { throw TrOCRError.resourceMissing(size) }
        let (encoder, decoder, tokenizer) = try await ModelCache.shared.models(for: size, resourceRoot: root)

        let pixelValues = try preprocess(image: image)
        let encoderOutput = try await encoder.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "pixel_values": MLFeatureValue(multiArray: pixelValues),
        ]))
        guard let encoderHidden = encoderOutput.featureValue(for: "encoder_hidden_states")?.multiArrayValue else {
            throw TrOCRError.predictionFailed
        }

        var ids = [tokenizer.decoderStartId]
        while ids.count < maxLength {
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

    /// Mirrors the HF `ViTImageProcessor`/`DeiTImageProcessor` preprocessing
    /// used by all three `-printed` checkpoints: resize to 384x384 (no aspect
    /// crop), normalize with mean=std=0.5 per channel.
    private static func preprocess(image: CGImage, size: Int = 384) throws -> MLMultiArray {
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
            if !FileManager.default.fileExists(atPath: compiledURL.path) {
                let freshlyCompiled = try MLModel.compileModel(at: packageURL)
                if FileManager.default.fileExists(atPath: compiledURL.path) {
                    try FileManager.default.removeItem(at: compiledURL)
                }
                try FileManager.default.moveItem(at: freshlyCompiled, to: compiledURL)
            }
            return try MLModel(contentsOf: compiledURL)
        }
    }
}
