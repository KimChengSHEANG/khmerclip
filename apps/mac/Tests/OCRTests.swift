import XCTest
import AppKit
// TesseractEngine.swift is compiled directly into this test bundle (see project.yml).

/// Image -> text regression test. Each `.png` in Tests/Fixtures/ocr/ is OCR'd and
/// compared (whitespace-normalized) against its sibling `.txt`. To add a case,
/// drop a `foo.png` + `foo.txt` into that folder — the suite discovers pairs, no
/// code change. Regenerate a baseline with:
///   ./Vendor/tesseract/bin/tesseract foo.png stdout \
///       --tessdata-dir Vendor/tesseract/tessdata -l khm+eng --psm 6
final class OCRTests: XCTestCase {
    /// Repo root derived from this file's path: Tests/OCRTests.swift -> ../ .
    private static let repoRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()

    override class func setUp() {
        // The vendored binary + tessdata live here; app uses Bundle.main, tests can't.
        TesseractEngine.resourceRootOverride = repoRoot.appendingPathComponent("Vendor/tesseract")
    }

    func testFixturesRecognizeToExpectedText() async throws {
        let dir = Self.repoRoot.appendingPathComponent("Tests/Fixtures/ocr")
        let pngs = try FileManager.default
            .contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "png" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        XCTAssertFalse(pngs.isEmpty, "No OCR fixtures found in \(dir.path)")

        for png in pngs {
            let expectedURL = png.deletingPathExtension().appendingPathExtension("txt")
            let expected = normalize(try String(contentsOf: expectedURL, encoding: .utf8))

            let image = try XCTUnwrap(loadCGImage(png), "Could not decode \(png.lastPathComponent)")
            let recognized = normalize(try await TesseractEngine.recognizeText(in: image))

            XCTAssertEqual(recognized, expected, "OCR mismatch for \(png.lastPathComponent)")
        }
    }

    private func normalize(_ s: String) -> String {
        s.components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }

    private func loadCGImage(_ url: URL) -> CGImage? {
        NSImage(contentsOf: url)?.cgImage(forProposedRect: nil, context: nil, hints: nil)
    }
}
