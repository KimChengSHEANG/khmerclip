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

    func testCrashSignatureIsSingleLineAndCapped() {
        let long = "NSInvalidArgumentException: " + String(repeating: "ក្សា\n", count: 500)
        let sig = Report.crash(signature: long, details: "d", engine: "e")["signature"] as? String ?? ""
        XCTAssertFalse(sig.contains("\n"))
        XCTAssertLessThanOrEqual(sig.unicodeScalars.count, 200)
        XCTAssertTrue(sig.hasPrefix("NSInvalidArgumentException:"))
    }

    // MARK: - CrashReporter

    private func payloads() -> [[String: Any]] {
        files().compactMap {
            (try? JSONSerialization.jsonObject(with: Data(contentsOf: $0))) as? [String: Any]
        }
    }

    private func writePending(_ text: String) throws -> URL {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("crash-pending.txt")
        try text.write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    func testCollectTurnsPendingFileIntoOneCrash() throws {
        let pending = try writePending("Signal 11\n0 KhmerClip \(NSHomeDirectory())/x")
        CrashReporter.collect(pendingURL: pending, outbox: outbox, engine: "tesseract", enabled: true)
        let all = payloads()
        XCTAssertEqual(all.count, 1)
        XCTAssertEqual(all.first?["type"] as? String, "crash")
        XCTAssertEqual(all.first?["signature"] as? String, "Signal 11")
        let details = all.first?["details"] as? String ?? ""
        XCTAssertFalse(details.contains(NSHomeDirectory()))
        XCTAssertTrue(details.contains("~/x"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: pending.path))
    }

    func testCollectDisabledDropsCrashes() throws {
        try outbox.enqueue(Report.crash(signature: "old", details: "d", engine: "e"))
        try queueFeedback()
        let pending = try writePending("Signal 6\nframes")
        CrashReporter.collect(pendingURL: pending, outbox: outbox, engine: "tesseract", enabled: false)
        let all = payloads()
        XCTAssertEqual(all.count, 1)
        XCTAssertEqual(all.first?["type"] as? String, "feedback")
        XCTAssertFalse(FileManager.default.fileExists(atPath: pending.path))
    }

    func testCollectIgnoresMissingOrEmptyPending() throws {
        let missing = dir.appendingPathComponent("none.txt")
        CrashReporter.collect(pendingURL: missing, outbox: outbox, engine: "e", enabled: true)
        let empty = try writePending("")
        CrashReporter.collect(pendingURL: empty, outbox: outbox, engine: "e", enabled: true)
        XCTAssertTrue(payloads().isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath: empty.path))
    }

    // MARK: - Crash signatures

    private let signalLog = """
        Signal 11
        0   KhmerClip                           0x0000000100a1c75c $s9KhmerClip13CrashReporterO7install10pendingURLy10Foundation3URLV_tFZySicfU0_To + 120
        1   libsystem_platform.dylib            0x000000019a1b2584 _sigtramp + 56
        2   AppKit                              0x000000019d001234 -[NSApplication run] + 400
        3   KhmerClip                           0x0000000100a10040 $s9KhmerClip18CaptureCoordinatorC6handleyyF + 36
        4   KhmerClip                           0x0000000100a10100 main + 12
        """

    func testSignalSignatureNamesFirstAppFrameOutsideTheHandler() {
        XCTAssertEqual(CrashReporter.signature(fromPending: signalLog),
                       "Signal 11 at $s9KhmerClip18CaptureCoordinatorC6handleyyF")
    }

    func testSignalSignatureFallsBackWithoutAppSymbol() {
        let stripped = """
            Signal 6
            0   KhmerClip                           0x0000000100a1c75c __mh_execute_header + 120
            1   libsystem_platform.dylib            0x000000019a1b2584 _sigtramp + 56
            2   KhmerClip                           0x0000000100a10040 __mh_execute_header + 9000
            """
        XCTAssertEqual(CrashReporter.signature(fromPending: stripped), "Signal 6")
        XCTAssertEqual(CrashReporter.signature(fromPending: "Signal 4\n"), "Signal 4")
        let onlyOthers = "Signal 10\n0   AppKit   0x000000019d001234 -[NSApplication run] + 400"
        XCTAssertEqual(CrashReporter.signature(fromPending: onlyOthers), "Signal 10")
    }

    func testExceptionSignatureDropsPerRunTokens() {
        let a = "NSInvalidArgumentException: -[NSNull length]: unrecognized selector sent to instance 0x600001c2a0f0\n0 CoreFoundation"
        let b = "NSInvalidArgumentException: -[NSNull length]: unrecognized selector sent to instance 0x7fd3b8e04a10\n0 CoreFoundation"
        XCTAssertEqual(CrashReporter.signature(fromPending: a), CrashReporter.signature(fromPending: b))
        XCTAssertFalse(CrashReporter.signature(fromPending: a).contains("0x6"))
        XCTAssertTrue(CrashReporter.signature(fromPending: a).hasPrefix("NSInvalidArgumentException: -[NSNull length]"))
        let c = "NSRangeException: *** index 5 beyond bounds [0 .. 3]"
        let d = "NSRangeException: *** index 9 beyond bounds [0 .. 7]"
        XCTAssertEqual(CrashReporter.signature(fromPending: c), CrashReporter.signature(fromPending: d))
    }

    func testCollectUsesParsedSignature() throws {
        let pending = try writePending(signalLog)
        CrashReporter.collect(pendingURL: pending, outbox: outbox, engine: "tesseract", enabled: true)
        XCTAssertEqual(payloads().first?["signature"] as? String,
                       "Signal 11 at $s9KhmerClip18CaptureCoordinatorC6handleyyF")
    }

    // MARK: - Crash dedup and cap

    private func crashCount() -> Int { payloads().filter { $0["type"] as? String == "crash" }.count }

    func testEnqueueCrashSkipsASignatureAlreadyQueued() throws {
        let home = NSHomeDirectory()
        XCTAssertNotNil(try outbox.enqueueCrash(Report.crash(signature: "Signal 11 at \(home)/x", details: "a", engine: "e")))
        XCTAssertNil(try outbox.enqueueCrash(Report.crash(signature: "Signal 11 at \(home)/x", details: "b", engine: "e")))
        XCTAssertNotNil(try outbox.enqueueCrash(Report.crash(signature: "Signal 6", details: "c", engine: "e")))
        XCTAssertEqual(crashCount(), 2)
    }

    func testEnqueueCrashCapsQueuedCrashes() throws {
        try queueFeedback()  // not a crash: not counted
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try "{not json".write(to: dir.appendingPathComponent("junk.json"), atomically: true, encoding: .utf8)
        for i in 0..<ReportOutbox.maxQueuedCrashes {
            XCTAssertNotNil(try outbox.enqueueCrash(Report.crash(signature: "E\(i)", details: "d", engine: "e")))
        }
        XCTAssertNil(try outbox.enqueueCrash(Report.crash(signature: "one too many", details: "d", engine: "e")))
        XCTAssertEqual(ReportOutbox.maxQueuedCrashes, 20)
        XCTAssertEqual(crashCount(), 20)
    }

    func testCollectSkipsACrashAlreadyQueued() throws {
        for _ in 0..<2 {
            let pending = try writePending(signalLog)
            CrashReporter.collect(pendingURL: pending, outbox: outbox, engine: "tesseract", enabled: true)
            XCTAssertFalse(FileManager.default.fileExists(atPath: pending.path))
        }
        XCTAssertEqual(crashCount(), 1)
    }
}
