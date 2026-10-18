import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// Builds anonymous report payloads in the relay's wire format (mirrors desktop reports.py).
enum Report {
    static let maxPNGBytes = 2_000_000      // relay limit
    static let targetPNGBytes = 1_500_000   // keeps the base64 body under the relay's 3 MB cap
    static let maxTextChars = 20_000
    private static let fieldChars = 100

    static func scrubHome(_ s: String) -> String {
        s.replacingOccurrences(of: NSHomeDirectory(), with: "~")
    }

    private static func line(_ s: String, _ limit: Int = fieldChars) -> String {
        scalarPrefix(s.split(whereSeparator: \.isWhitespace).joined(separator: " "), limit)
    }

    /// The relay's limits count UTF-16 code units, and Khmer clusters span several scalars,
    /// so caps count Unicode scalars rather than grapheme clusters.
    private static func scalarPrefix(_ s: String, _ n: Int) -> String {
        String(String.UnicodeScalarView(s.unicodeScalars.prefix(n)))
    }

    private static func cap(_ s: String) -> String { scalarPrefix(s, maxTextChars) }

    private static func base(_ engine: String, type: String) -> [String: Any] {
        let version = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "dev"
        let eng = line(engine)
        return [
            "type": type,
            "id": UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased(),
            "app_version": line(version),
            "os": line("macOS " + ProcessInfo.processInfo.operatingSystemVersionString),
            "engine": eng.isEmpty ? "unknown" : eng,
        ]
    }

    static func capture(pngURL: URL, meta: [String: Any], ocrText: String, correction: String,
                        engine: String) -> [String: Any] {
        var p = base(engine, type: "capture")
        p["image_png_b64"] = pngWithinLimit(pngURL).base64EncodedString()
        p["ocr_text"] = cap(ocrText)
        p["correction"] = cap(correction)
        p["meta"] = meta
        return p
    }

    static func feedback(message: String, email: String, engine: String) -> [String: Any] {
        var p = base(engine, type: "feedback")
        p["message"] = cap(message)
        let trimmed = email.trimmingCharacters(in: .whitespacesAndNewlines)
        p["email"] = trimmed.count > 254 ? "" : trimmed  // an over-long email must not get the whole report rejected
        return p
    }

    static func crash(signature: String, details: String, engine: String) -> [String: Any] {
        var p = base(engine, type: "crash")
        p["signature"] = scalarPrefix(scrubHome(line(signature, 1_000_000)), 200)
        p["details"] = cap(scrubHome(details))
        return p
    }

    // MARK: - PNG downscaling

    private static func encodePNG(_ image: CGImage) -> Data? {
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil)
        else { return nil }
        CGImageDestinationAddImage(dest, image, nil)
        return CGImageDestinationFinalize(dest) ? data as Data : nil
    }

    private static func halved(_ image: CGImage) -> CGImage? {
        let w = max(1, image.width / 2), h = max(1, image.height / 2)
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return nil }
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        return ctx.makeImage()
    }

    /// PNG bytes of the file, halved repeatedly until they fit `targetPNGBytes`.
    private static func pngWithinLimit(_ url: URL) -> Data {
        let original = (try? Data(contentsOf: url)) ?? Data()
        if original.count <= targetPNGBytes { return original }
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              var image = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return original }
        while true {
            guard let data = encodePNG(image) else { return original }
            if data.count <= targetPNGBytes || min(image.width, image.height) <= 1 { return data }
            guard let smaller = halved(image) else { return data }
            image = smaller
        }
    }
}
