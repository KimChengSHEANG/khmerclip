import Foundation

/// Post-processes OCR output according to the Preserve Linebreaks preference.
/// When disabled, visual wrapping from the source is flattened into a paragraph.
enum OCRTextFormatter {
    static func apply(_ text: String, preserveLinebreaks: Bool) -> String {
        guard !preserveLinebreaks else { return text }
        return text
            .split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }
}
