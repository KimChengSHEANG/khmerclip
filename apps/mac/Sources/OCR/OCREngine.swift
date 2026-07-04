import AppKit

/// Which text-recognition backend CaptureCoordinator dispatches to. Tesseract
/// handles Khmer+English reliably via the vendored binary; TrOCR runs a
/// CoreML model (see TrOCRModelSize) -- small/base/large are English-only,
/// and the khmer model is a Khmer printed-text fine-tune (see TrOCREngine's
/// doc comment).
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
/// (lkhapple/Khmer-TrOCR-OCR) -- not a size tier, but a
/// distinct model kept alongside them since Khmer support doesn't fit the
/// small/base/large scale.
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
        progress: (@MainActor (OCRProgress) -> Void)? = nil
    ) async throws -> String {
        switch engine {
        case .tesseract:
            return try await TesseractEngine.recognizeText(in: image)
        case .trocr:
            return try await TrOCREngine.recognizeText(in: image, model: trocrModel, progress: progress)
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
