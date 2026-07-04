import AppKit

/// Which text-recognition backend CaptureCoordinator dispatches to. Tesseract
/// handles Khmer+English via the vendored binary; TrOCR runs a CoreML model
/// converted from a `microsoft/trocr-*-printed` checkpoint (English only until
/// the model is fine-tuned for Khmer).
enum OCREngineKind: String, CaseIterable, Identifiable {
    case tesseract
    case trocr

    var id: String { rawValue }
    var displayName: String {
        switch self {
        case .tesseract: return "Tesseract"
        case .trocr: return "TrOCR"
        }
    }
}

/// Which size of the TrOCR model to run. Larger models are more accurate but
/// slower and bigger on disk.
enum TrOCRModelSize: String, CaseIterable, Identifiable {
    case small
    case base
    case large

    var id: String { rawValue }
    var displayName: String {
        switch self {
        case .small: return "Small"
        case .base: return "Base"
        case .large: return "Large"
        }
    }
}

protocol OCREngine {
    static func recognizeText(in image: CGImage) async throws -> String
}

extension TesseractEngine: OCREngine {}

enum OCRDispatcher {
    static func recognizeText(in image: CGImage, engine: OCREngineKind, trocrModel: TrOCRModelSize) async throws -> String {
        switch engine {
        case .tesseract:
            return try await TesseractEngine.recognizeText(in: image)
        case .trocr:
            return try await TrOCREngine.recognizeText(in: image, model: trocrModel)
        }
    }
}
