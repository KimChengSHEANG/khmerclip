import Vision
import AppKit

enum OCREngineError: Error, LocalizedError {
    case noSupportedLanguage([String])

    var errorDescription: String? {
        switch self {
        case .noSupportedLanguage(let requested):
            return "None of the selected languages (\(requested.joined(separator: ", "))) are supported by this recognizer."
        }
    }
}

/// Vision-based recognizer. Vision does not support Khmer (verified against
/// VNRecognizeTextRequest.supportedRecognitionLanguages / RecognizeTextRequest
/// on macOS 26) — Khmer capture is routed through TesseractEngine instead, see
/// TextRecognizer.
enum OCREngine {
    static func supportedLanguages(from requested: [String]) throws -> [String] {
        let supported = (try? VNRecognizeTextRequest.supportedRecognitionLanguages(for: .accurate, revision: VNRecognizeTextRequestRevision3)) ?? []
        let filtered = requested.filter { supported.contains($0) }
        guard !filtered.isEmpty else { throw OCREngineError.noSupportedLanguage(requested) }
        return filtered
    }

    static func recognizeText(in image: CGImage, languages: [String]) async throws -> String {
        let resolvedLanguages = try supportedLanguages(from: languages)
        return try await withCheckedThrowingContinuation { continuation in
            let request = VNRecognizeTextRequest { request, error in
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                let observations = request.results as? [VNRecognizedTextObservation] ?? []
                let lines = observations.compactMap { $0.topCandidates(1).first?.string }
                continuation.resume(returning: lines.joined(separator: "\n"))
            }
            request.recognitionLevel = .accurate
            request.usesLanguageCorrection = true
            request.recognitionLanguages = resolvedLanguages

            let handler = VNImageRequestHandler(cgImage: image, options: [:])
            do {
                try handler.perform([request])
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }
}
