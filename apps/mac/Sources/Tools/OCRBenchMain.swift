import AppKit
import Foundation

@main
struct OCRBenchMain {
    static func main() async {
        do {
            let arguments = CommandLine.arguments
            guard arguments.count == 4, arguments[1] == "--engine" else {
                throw CLIError.usage
            }

            let engine = arguments[2]
            guard engine == "trocr" || engine == "tesseract" else {
                throw CLIError.unsupportedEngine(engine)
            }

            let vendorEnv = ProcessInfo.processInfo.environment["KHMERCLIP_VENDOR"]
            let vendor = URL(fileURLWithPath: vendorEnv ?? "Vendor", isDirectory: true)
            TesseractEngine.resourceRootOverride = vendor.appendingPathComponent(
                "tesseract",
                isDirectory: true
            )
            TrOCREngine.resourceRootOverride = vendor.appendingPathComponent(
                "trocr",
                isDirectory: true
            )

            let imageURL = URL(fileURLWithPath: arguments[3])
            guard let image = NSImage(contentsOf: imageURL),
                  let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
                throw CLIError.imageUnreadable(imageURL.path)
            }

            let started = ContinuousClock.now
            let text: String
            if engine == "tesseract" {
                text = try await TesseractEngine.recognizeText(in: cgImage)
            } else {
                text = try await TrOCREngine.recognizeText(in: cgImage, model: .khmer)
            }
            let elapsed = started.duration(to: .now)
            let milliseconds = Double(elapsed.components.seconds) * 1_000
                + Double(elapsed.components.attoseconds) / 1_000_000_000_000_000

            fputs(String(format: "ms=%.3f\n", milliseconds), stderr)
            print(text)
        } catch {
            fputs("\(error)\n", stderr)
            exit(EXIT_FAILURE)
        }
    }
}

private enum CLIError: LocalizedError, CustomStringConvertible {
    case usage
    case unsupportedEngine(String)
    case imageUnreadable(String)

    var description: String {
        switch self {
        case .usage:
            return "usage: OCRBench --engine trocr|tesseract <image.png>"
        case .unsupportedEngine(let engine):
            return "unsupported engine: \(engine)"
        case .imageUnreadable(let path):
            return "could not load image: \(path)"
        }
    }

    var errorDescription: String? { description }
}
