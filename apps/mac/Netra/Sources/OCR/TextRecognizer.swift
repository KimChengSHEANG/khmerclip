import AppKit

/// Picks the right OCR backend per requested language set. Vision covers the
/// languages Apple ships models for; Tesseract (vendored, see Vendor/tesseract)
/// covers Khmer, which Vision cannot recognize at all as of macOS 26.
enum TextRecognizer {
    static func recognizeText(in image: CGImage, languages: [String]) async throws -> String {
        if languages.contains(where: { TesseractEngine.tesseractLanguages(from: [$0]).isEmpty == false }) {
            return try await TesseractEngine.recognizeText(in: image, languages: languages)
        }
        return try await OCREngine.recognizeText(in: image, languages: languages)
    }
}
