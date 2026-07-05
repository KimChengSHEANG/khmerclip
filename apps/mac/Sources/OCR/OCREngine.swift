import AppKit

/// Which text-recognition backend CaptureCoordinator dispatches to. Tesseract
/// handles Khmer+English reliably via the vendored binary; TrOCR runs a
/// CoreML Khmer fine-tune (see TrOCRModelSize); EasyOCR runs a Python
/// EasyOCR stack with a custom Khmer recognition model (see EasyOCRModelSize).
enum OCREngineKind: String, CaseIterable, Identifiable {
    case tesseract
    case trocr
    case easyocr

    var id: String { rawValue }
    var displayName: String {
        switch self {
        case .tesseract: return "Tesseract"
        case .trocr: return "TrOCR"
        case .easyocr: return "EasyOCR"
        }
    }
}

/// Which TrOCR model to run. khmer is a Khmer printed-text fine-tune
/// (lkhapple/Khmer-TrOCR-OCR); the English small/base/large size tiers were
/// dropped since this app targets Khmer OCR.
enum TrOCRModelSize: String, CaseIterable, Identifiable {
    case khmer

    var id: String { rawValue }
    var displayName: String {
        switch self {
        case .khmer: return "Khmer"
        }
    }
}

/// Which EasyOCR recognition model to run. khmer uses a custom fine-tuned
/// network (models/khmer.pth + khmer/user_network/{khmer.yaml,khmer.py}).
/// EasyOCR has no built-in Khmer support; install your model with
/// Vendor/easyocr/install_khmer_model.py when ready.
enum EasyOCRModelSize: String, CaseIterable, Identifiable {
    case khmer

    var id: String { rawValue }
    var displayName: String {
        switch self {
        case .khmer: return "Khmer"
        }
    }
}

protocol OCREngine {
    static func recognizeText(in image: CGImage) async throws -> String
}

extension TesseractEngine: OCREngine {}

/// A phase + optional completion fraction for the OCR progress box.
/// `fraction == nil` means indeterminate (spinner); otherwise 0...1.
struct OCRProgress {
    var label: String
    var fraction: Double?
}

enum OCRDispatcher {
    static func recognizeText(
        in image: CGImage,
        engine: OCREngineKind,
        trocrModel: TrOCRModelSize,
        easyocrModel: EasyOCRModelSize,
        progress: (@MainActor (OCRProgress) -> Void)? = nil
    ) async throws -> String {
        switch engine {
        case .tesseract:
            return try await TesseractEngine.recognizeText(in: image)
        case .trocr:
            return try await TrOCREngine.recognizeText(in: image, model: trocrModel, progress: progress)
        case .easyocr:
            return try await EasyOCREngine.recognizeText(in: image, model: easyocrModel, progress: progress)
        }
    }
}

extension OCRProgress {
    static let preparing = OCRProgress(label: "Preparing…", fraction: nil)
    static let loadingModel = OCRProgress(label: "Loading model…", fraction: nil)
    static let encoding = OCRProgress(label: "Encoding image…", fraction: nil)
    // Decode length is unknown until EOS, so a completion fraction would be a lie
    // (it caps at maxLength and jumps to done). Show token count as live evidence
    // of work instead, with an indeterminate bar.
    static func reading(tokens: Int) -> OCRProgress {
        OCRProgress(label: "Reading text… (\(tokens))", fraction: nil)
    }
}
