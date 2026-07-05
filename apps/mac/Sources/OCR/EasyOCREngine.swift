import AppKit

enum EasyOCRError: Error, LocalizedError {
    case bundleMissing
    case pythonMissing
    case pngEncodingFailed
    case modelMissing(EasyOCRModelSize)
    case processFailed(String)

    var errorDescription: String? {
        switch self {
        case .bundleMissing:
            return "EasyOCR resources are missing from the app bundle."
        case .pythonMissing:
            return "Python is not available. Run `make easyocr-venv` in the repo to set up EasyOCR."
        case .pngEncodingFailed:
            return "Could not encode the capture as PNG for EasyOCR."
        case .modelMissing(let size):
            return "EasyOCR \(size.displayName) model is not installed. Add khmer.pth to Vendor/easyocr/models/ and khmer.yaml + khmer.py to Vendor/easyocr/khmer/user_network/, then run install_khmer_model.py."
        case .processFailed(let message):
            return message
        }
    }
}

/// Shells out to Vendor/easyocr/recognize.py (EasyOCR + PyTorch) for text
/// recognition. Uses CRAFT for detection (Vendor/easyocr/models/, downloaded
/// via `make easyocr-models`) and a custom Khmer recognition network
/// (Vendor/easyocr/khmer/user_network/ + models/khmer.pth). EasyOCR does not ship Khmer
/// out of the box; drop in your fine-tuned khmer.{pth,yaml,py} with
/// install_khmer_model.py when ready.
enum EasyOCREngine {
    /// Points at the folder containing recognize.py, models/, and khmer/.
    /// Normally the app's bundled `easyocr` resource; tests set this to the
    /// repo's `Vendor/easyocr` so OCR can run in a hostless test bundle.
    static var resourceRootOverride: URL?

    private static var resourceRoot: URL? {
        resourceRootOverride
            ?? Bundle.main.resourceURL?.appendingPathComponent("easyocr", isDirectory: true)
    }

    /// Warms PyTorch + model weights while the user drags a selection. Best-effort:
    /// errors are swallowed and surfaced on the real recognizeText call.
    static func preload(model size: EasyOCRModelSize) async {
        guard let root = resourceRoot else { return }
        guard modelFilesExist(for: size, root: root) else { return }
        _ = try? await runRecognize(
            root: root,
            model: size,
            imageURL: nil,
            warmup: true
        )
    }

    static func recognizeText(
        in image: CGImage,
        model size: EasyOCRModelSize,
        progress: (@MainActor (OCRProgress) -> Void)? = nil
    ) async throws -> String {
        guard let root = resourceRoot else { throw EasyOCRError.bundleMissing }
        guard modelFilesExist(for: size, root: root) else {
            throw EasyOCRError.modelMissing(size)
        }

        await progress?(.loadingModel)
        let imageURL = try writeTemporaryPNG(image)
        defer { try? FileManager.default.removeItem(at: imageURL) }

        await progress?(.encoding)
        return try await runRecognize(root: root, model: size, imageURL: imageURL, warmup: false)
    }

    private static func modelFilesExist(for size: EasyOCRModelSize, root: URL) -> Bool {
        let name = size.rawValue
        let weights = root
            .appendingPathComponent("models/\(name).pth")
        let yaml = root
            .appendingPathComponent(name, isDirectory: true)
            .appendingPathComponent("user_network/\(name).yaml")
        let py = root
            .appendingPathComponent(name, isDirectory: true)
            .appendingPathComponent("user_network/\(name).py")
        let fm = FileManager.default
        return fm.isReadableFile(atPath: weights.path)
            && fm.isReadableFile(atPath: yaml.path)
            && fm.isReadableFile(atPath: py.path)
    }

    private static func pythonBinary(in root: URL) -> URL? {
        let candidates = [
            root.appendingPathComponent("venv/bin/python3"),
            URL(fileURLWithPath: "/opt/homebrew/bin/python3"),
            URL(fileURLWithPath: "/usr/local/bin/python3"),
            URL(fileURLWithPath: "/usr/bin/python3"),
        ]
        for url in candidates where FileManager.default.isExecutableFile(atPath: url.path) {
            return url
        }
        return nil
    }

    private static func runRecognize(
        root: URL,
        model size: EasyOCRModelSize,
        imageURL: URL?,
        warmup: Bool
    ) async throws -> String {
        guard let python = pythonBinary(in: root) else { throw EasyOCRError.pythonMissing }
        let script = root.appendingPathComponent("recognize.py")
        guard FileManager.default.isReadableFile(atPath: script.path) else {
            throw EasyOCRError.bundleMissing
        }

        var arguments = [
            script.path,
            "--resource-root", root.path,
            "--model", size.rawValue,
        ]
        if warmup {
            arguments.append("--warmup")
        } else if let imageURL {
            arguments.append(imageURL.path)
        }

        return try await withCheckedThrowingContinuation { continuation in
            let process = Process()
            process.executableURL = python
            process.arguments = arguments
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
                } else if proc.terminationStatus == 2 {
                    let message = String(data: errData, encoding: .utf8) ?? "EasyOCR model is not installed."
                    continuation.resume(throwing: EasyOCRError.processFailed(message))
                } else {
                    let message = String(data: errData, encoding: .utf8)
                        ?? String(data: outData, encoding: .utf8)
                        ?? "EasyOCR exited with \(proc.terminationStatus)"
                    continuation.resume(throwing: EasyOCRError.processFailed(message))
                }
            }
        }
    }

    private static func writeTemporaryPNG(_ image: CGImage) throws -> URL {
        let rep = NSBitmapImageRep(cgImage: image)
        guard let data = rep.representation(using: .png, properties: [:]) else {
            throw EasyOCRError.pngEncodingFailed
        }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".png")
        try data.write(to: url)
        return url
    }
}
