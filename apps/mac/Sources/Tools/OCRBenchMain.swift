import AppKit
import Foundation

@main
struct OCRBenchMain {
    static func main() async {
        do {
            let arguments = CommandLine.arguments
            guard arguments.count >= 4, arguments[1] == "--engine" else {
                throw CLIError.usage
            }

            let engine = arguments[2]
            guard engine == "trocr" || engine == "tesseract" else {
                throw CLIError.unsupportedEngine(engine)
            }
            var imagePaths = Array(arguments.dropFirst(3))
            let shouldPreload = imagePaths.first == "--preload"
            if shouldPreload {
                imagePaths.removeFirst()
            }
            guard !imagePaths.isEmpty else {
                throw CLIError.usage
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

            if engine == "trocr" && (shouldPreload || imagePaths.count > 1) {
                await TrOCREngine.preload(model: .khmer)
            }

            for imagePath in imagePaths {
                let imageURL = URL(fileURLWithPath: imagePath)
                guard let image = NSImage(contentsOf: imageURL),
                      let cgImage = image.cgImage(
                        forProposedRect: nil,
                        context: nil,
                        hints: nil
                      ) else {
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
                if engine == "trocr" {
                    let s = TrOCREngine.lastRunStats
                    fputs(String(format: "stats strips=%d enc_ms=%.1f steps=%d dec_ms=%.1f\n",
                                 s.strips, s.encoderMs, s.decoderSteps, s.decoderMs), stderr)
                }
                let oneLine = text.split(whereSeparator: \.isNewline).joined(separator: " ")
                print(oneLine)
            }
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
            return "usage: OCRBench --engine trocr|tesseract [--preload] <image.png> [image.png ...]"
        case .unsupportedEngine(let engine):
            return "unsupported engine: \(engine)"
        case .imageUnreadable(let path):
            return "could not load image: \(path)"
        }
    }

    var errorDescription: String? { description }
}
