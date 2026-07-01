import Vision
import AppKit

enum OCREngine {
    /// Languages we ask Vision to recognize, filtered down to whatever the
    /// installed OS build actually supports (Khmer support varies by macOS version).
    static func supportedLanguages(from requested: [String]) -> [String] {
        let supported = (try? VNRecognizeTextRequest.supportedRecognitionLanguages(for: .accurate, revision: VNRecognizeTextRequestRevision3)) ?? []
        let filtered = requested.filter { supported.contains($0) }
        return filtered.isEmpty ? supported : filtered
    }

    static func recognizeText(in image: CGImage, languages: [String]) async throws -> String {
        try await withCheckedThrowingContinuation { continuation in
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
            request.recognitionLanguages = supportedLanguages(from: languages)

            let handler = VNImageRequestHandler(cgImage: image, options: [:])
            do {
                try handler.perform([request])
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }
}
