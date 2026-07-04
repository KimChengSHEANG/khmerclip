import AppKit

/// Which text-recognition backend CaptureCoordinator dispatches to. Tesseract
/// handles Khmer+English reliably via the vendored binary; TrOCR runs a
/// CoreML model (see TrOCRModelSize) -- small/base/large are English-only,
/// and the khmer model is an early, unreliable Khmer fine-tune (see
/// TrOCREngine's doc comment).
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

/// Which TrOCR model to run. small/base/large are the official
/// microsoft/trocr-*-printed checkpoints (larger = more accurate, slower,
/// bigger on disk); khmer is a separate Khmer-capable fine-tune
/// (KiteAether/khmer-trcor-06-10) -- not a size tier, but a distinct model
/// kept alongside them since Khmer support doesn't fit the small/base/large
/// scale.
enum TrOCRModelSize: String, CaseIterable, Identifiable {
    case small
    case base
    case large
    case khmer

    var id: String { rawValue }
    var displayName: String {
        switch self {
        case .small: return "Small"
        case .base: return "Base"
        case .large: return "Large"
        case .khmer: return "Khmer"
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
