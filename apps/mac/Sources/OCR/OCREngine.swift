import AppKit

/// Which text-recognition backend CaptureCoordinator dispatches to. Tesseract
/// handles Khmer+English reliably via the vendored binary; TrOCR runs a
/// CoreML Khmer fine-tune (see TrOCRModelSize).
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

protocol OCREngine {
    static func recognizeText(in image: CGImage) async throws -> String
}

extension TesseractEngine: OCREngine {}

/// A phase + optional completion fraction for the OCR progress box.
/// `fraction == nil` means indeterminate (spinner); otherwise 0...1.
struct OCRProgress {
    var label: String
    var fraction: Double?
    var symbol: String = "text.viewfinder"
    /// Secondary live value (e.g. token count) shown as a badge beside the label.
    var detail: String? = nil
}

enum OCRDispatcher {
    static func recognizeText(
        in image: CGImage,
        engine: OCREngineKind,
        progress: (@MainActor (OCRProgress) -> Void)? = nil
    ) async throws -> String {
        switch engine {
        case .tesseract:
            return try await TesseractEngine.recognizeText(in: image)
        case .trocr:
            return try await TrOCREngine.recognizeText(in: image, model: .khmer, progress: progress)
        }
    }
}

extension OCRProgress {
    static let preparing = OCRProgress(label: "Preparing…", fraction: nil, symbol: "hourglass")
    static let loadingModel = OCRProgress(label: "Loading model…", fraction: nil, symbol: "cpu")
    static let encoding = OCRProgress(label: "Encoding image…", fraction: nil, symbol: "photo")
    // Decode length is unknown until EOS, so a completion fraction would be a lie
    // (it caps at maxLength and jumps to done). Show token count as live evidence
    // of work instead, with an indeterminate bar.
    static func reading(tokens: Int) -> OCRProgress {
        OCRProgress(label: "Reading text…", fraction: nil, symbol: "text.viewfinder", detail: "\(tokens)")
    }
}
