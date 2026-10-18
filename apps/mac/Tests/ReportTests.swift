import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import XCTest

final class ReportTests: XCTestCase {
    private var dir: URL!
    private let relay = URL(string: "https://relay.test/report")!

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory.appendingPathComponent("report-tests-\(UUID().uuidString)")
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: dir)
    }

    private var outbox: ReportOutbox { ReportOutbox(dir: dir) }

    private func files() -> [URL] {
        ((try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension == "json" }
    }

    private func queueFeedback() throws -> URL {
        try outbox.enqueue(Report.feedback(message: "hi", email: "", engine: "tesseract"))
    }

    /// Collects posted payloads from the (non-Sendable) closure safely.
    private final class Box: @unchecked Sendable {
        var posted: [[String: Any]] = []
    }

    func testEnqueueAndFlushDeletesOnSuccess() async throws {
        let file = try queueFeedback()
        let box = Box()
        let sent = await outbox.flush(url: relay) { payload in box.posted.append(payload); return 200 }
        XCTAssertEqual(sent, 1)
        XCTAssertFalse(FileManager.default.fileExists(atPath: file.path))
        XCTAssertEqual(box.posted.first?["type"] as? String, "feedback")
    }

    func testFlushKeepsFileOnServerError() async throws {
        let file = try queueFeedback()
        let sent = await outbox.flush(url: relay) { _ in 503 }
        XCTAssertEqual(sent, 0)
        XCTAssertTrue(FileManager.default.fileExists(atPath: file.path))
        let none = await outbox.flush(url: relay) { _ in 0 }
        XCTAssertEqual(none, 0)
        XCTAssertTrue(FileManager.default.fileExists(atPath: file.path))
    }

    func testFlushDropsRejected() async throws {
        for status in [400, 401, 413] {
            let file = try queueFeedback()
            let sent = await outbox.flush(url: relay) { _ in status }
            XCTAssertEqual(sent, 0)
            XCTAssertFalse(FileManager.default.fileExists(atPath: file.path), "status \(status)")
        }
    }

    func testFlushSkipsUnconfiguredURL() async throws {
        let file = try queueFeedback()
        let box = Box()
        let sent = await outbox.flush(url: ReportOutbox.unsetURL) { p in box.posted.append(p); return 200 }
        XCTAssertEqual(sent, 0)
        XCTAssertTrue(box.posted.isEmpty)
        XCTAssertTrue(FileManager.default.fileExists(atPath: file.path))
    }

    func testFlushRemovesUnreadableFiles() async throws {
        _ = try queueFeedback()
        let bad = dir.appendingPathComponent("bad.json")
        try Data("not json".utf8).write(to: bad)
        let box = Box()
        let sent = await outbox.flush(url: relay) { p in box.posted.append(p); return 200 }
        XCTAssertEqual(sent, 1)
        XCTAssertEqual(box.posted.count, 1)
        XCTAssertFalse(FileManager.default.fileExists(atPath: bad.path))
    }

    func testDropTypeRemovesOnlyThatType() throws {
        _ = try queueFeedback()
        _ = try outbox.enqueue(Report.crash(signature: "s", details: "d", engine: "e"))
        outbox.dropType("crash")
        XCTAssertEqual(files().count, 1)
    }

    func testBaseFieldsAreNormalized() {
        let p = Report.feedback(message: "m", email: "e@x.y", engine: "  multi\n line  ")
        XCTAssertEqual(p["engine"] as? String, "multi line")
        XCTAssertEqual(Report.feedback(message: "m", email: "", engine: " ")["engine"] as? String, "unknown")
        let long = Report.feedback(message: String(repeating: "x", count: 30_000), email: "",
                                   engine: String(repeating: "e", count: 300))
        XCTAssertEqual((long["engine"] as? String)?.count, 100)
        XCTAssertEqual((long["message"] as? String)?.count, 20_000)
        let id = p["id"] as? String ?? ""
        XCTAssertEqual(id.count, 32)
        XCTAssertEqual(id, id.lowercased())
        XCTAssertFalse(id.contains("-"))
    }

    func testCrashScrubsHome() {
        let home = NSHomeDirectory()
        let p = Report.crash(signature: "Crash at \(home)/a.swift", details: "frame \(home)/x/y.swift:3", engine: "e")
        let details = p["details"] as? String ?? ""
        XCTAssertFalse(details.contains(home))
        XCTAssertTrue(details.contains("~/"))
        XCTAssertFalse((p["signature"] as? String ?? "").contains(home))
        XCTAssertNil(p["image_png_b64"])
        XCTAssertEqual(p["type"] as? String, "crash")
    }

    func testCapturePayloadDownscalesLargeImage() throws {
        let w = 2400, h = 1600
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        for i in 0..<bytes.count { bytes[i] = UInt8.random(in: 0...255) }
        let ctx = CGContext(data: &bytes, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                            space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        let image = ctx.makeImage()!
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("noise.png")
        let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(dest, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(dest))
        XCTAssertGreaterThan(try Data(contentsOf: url).count, Report.maxPNGBytes)

        let p = Report.capture(pngURL: url, meta: ["k": "v"], ocrText: "t", correction: "", engine: "e")
        let b64 = try XCTUnwrap(p["image_png_b64"] as? String)
        let png = try XCTUnwrap(Data(base64Encoded: b64))
        XCTAssertLessThanOrEqual(png.count, Report.targetPNGBytes)
        XCTAssertEqual(Array(png.prefix(4)), [0x89, 0x50, 0x4E, 0x47])
        XCTAssertEqual(p["type"] as? String, "capture")
        XCTAssertEqual(p["correction"] as? String, "")
    }

    func testTextCapCountsUnicodeScalars() {
        let khmer = String(repeating: "ក្សា", count: 6000)  // 4 scalars per cluster
        let payload = Report.crash(signature: "s", details: khmer, engine: "e")
        let details = payload["details"] as? String ?? ""
        XCTAssertLessThanOrEqual(details.unicodeScalars.count, 20_000)
        XCTAssertGreaterThan(details.unicodeScalars.count, 0)
    }

    func testFeedbackEmailIsTrimmedAndBlankedWhenTooLong() {
        let ok = Report.feedback(message: "m", email: "  a@b.co \n", engine: "e")
        XCTAssertEqual(ok["email"] as? String, "a@b.co")
        let long = String(repeating: "a", count: 250) + "@b.co"  // 255 chars
        let bad = Report.feedback(message: "m", email: long, engine: "e")
        XCTAssertEqual(bad["email"] as? String, "")
    }
}
