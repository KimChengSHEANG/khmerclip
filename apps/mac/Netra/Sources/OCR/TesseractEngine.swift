import AppKit

enum TesseractError: Error {
    case bundleMissing
    case pngEncodingFailed
    case processFailed(String)
}

/// Wraps a vendored, self-contained Tesseract binary (see Vendor/tesseract) for
/// scripts Vision doesn't recognize — Khmer chief among them. The binary and its
/// dylibs were rewritten with `dylibbundler` to `@executable_path`-relative install
/// names so it runs without a Homebrew install on the end user's Mac.
enum TesseractEngine {
    /// Maps our BCP-47-ish preference codes to Tesseract's 3-letter language codes,
    /// limited to what we actually bundle traineddata for.
    private static let languageMap: [String: String] = [
        "km-KH": "khm",
        "en-US": "eng",
    ]

    static func tesseractLanguages(from requested: [String]) -> [String] {
        requested.compactMap { languageMap[$0] }
    }

    private static var resourceRoot: URL? {
        Bundle.main.resourceURL?.appendingPathComponent("tesseract", isDirectory: true)
    }

    static func recognizeText(in image: CGImage, languages: [String]) async throws -> String {
        guard let root = resourceRoot else { throw TesseractError.bundleMissing }
        let binary = root.appendingPathComponent("bin/tesseract")
        let tessdata = root.appendingPathComponent("tessdata")
        let langs = tesseractLanguages(from: languages)
        guard !langs.isEmpty else { throw TesseractError.bundleMissing }

        let imageURL = try writeTemporaryPNG(image)
        defer { try? FileManager.default.removeItem(at: imageURL) }

        return try await withCheckedThrowingContinuation { continuation in
            let process = Process()
            process.executableURL = binary
            process.arguments = [
                imageURL.path, "stdout",
                "--tessdata-dir", tessdata.path,
                "-l", langs.joined(separator: "+"),
                "--psm", "6",
            ]
            let outPipe = Pipe()
            let errPipe = Pipe()
            process.standardOutput = outPipe
            process.standardError = errPipe

            do {
                try process.run()
            } catch {
                continuation.resume(throwing: error)
                return
            }
            process.terminationHandler = { proc in
                let outData = outPipe.fileHandleForReading.readDataToEndOfFile()
                let errData = errPipe.fileHandleForReading.readDataToEndOfFile()
                if proc.terminationStatus == 0 {
                    let text = String(data: outData, encoding: .utf8) ?? ""
                    continuation.resume(returning: text.trimmingCharacters(in: .whitespacesAndNewlines))
                } else {
                    let message = String(data: errData, encoding: .utf8) ?? "tesseract exited with \(proc.terminationStatus)"
                    continuation.resume(throwing: TesseractError.processFailed(message))
                }
            }
        }
    }

    private static func writeTemporaryPNG(_ image: CGImage) throws -> URL {
        let rep = NSBitmapImageRep(cgImage: image)
        guard let data = rep.representation(using: .png, properties: [:]) else {
            throw TesseractError.pngEncodingFailed
        }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".png")
        try data.write(to: url)
        return url
    }
}
