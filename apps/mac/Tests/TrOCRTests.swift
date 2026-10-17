import XCTest
import AppKit
import CoreML
// TrOCREngine.swift + TrOCRTokenizer.swift are compiled directly into this test bundle (see project.yml).

final class TrOCRTokenizerTests: XCTestCase {
    /// Ids/expected strings recorded from the reference HF tokenizers
    /// (`RobertaTokenizer.decode` / `XLMRobertaTokenizer.decode`, both with
    /// `skip_special_tokens=True`) for the first few real vocab entries of
    /// trocr-base ('.', 'Ġthe', ',', 'Ġto' at ids 4,5,6,7) and trocr-small
    /// ('▁the', '.', ',', '▁to' at ids 4,5,6,7). Exercises the two pieces of
    /// decode logic this file actually implements: GPT2 byte-level BPE
    /// reconstruction (leading space kept) and SentencePiece '▁'-to-space
    /// substitution (leading space trimmed).
    func testBPEDecodeReconstructsBytesAndKeepsLeadingSpace() throws {
        let json = """
        {"kind":"bpe","bos_id":0,"eos_id":2,"pad_id":1,"unk_id":3,"decoder_start_id":2,
         "id_to_token":["<s>","<pad>","</s>","<unk>",".","Ġthe",",","Ġto"]}
        """
        let tokenizer = try TrOCRTokenizer(data: Data(json.utf8))
        XCTAssertEqual(tokenizer.decode(ids: [0, 5, 7, 2]), " the to")
    }

    /// The khmer fine-tune's vocab mixes byte-BPE pieces with literal added
    /// tokens (raw Khmer UTF-8). literal_ids marks the latter; decode must
    /// byte-map the former but pass the latter through verbatim. Here ids 4,5
    /// are byte-BPE 'a'/'b' and id 6 is a literal 'ក' (Khmer). Expect "abក".
    func testBPEDecodePassesLiteralAddedTokensThroughVerbatim() throws {
        let json = """
        {"kind":"bpe","bos_id":0,"eos_id":2,"pad_id":1,"unk_id":3,"decoder_start_id":2,
         "literal_ids":[6],
         "id_to_token":["<s>","<pad>","</s>","<unk>","a","b","ក"]}
        """
        let tokenizer = try TrOCRTokenizer(data: Data(json.utf8))
        XCTAssertEqual(tokenizer.decode(ids: [0, 4, 5, 6, 2]), "abក")
    }

    func testSentencePieceDecodeReplacesMarkerAndTrimsLeadingSpace() throws {
        let json = """
        {"kind":"sentencepiece","bos_id":0,"eos_id":2,"pad_id":1,"unk_id":3,"decoder_start_id":2,
         "id_to_token":["<s>","<pad>","</s>","<unk>","▁the",".",",","▁to"]}
        """
        let tokenizer = try TrOCRTokenizer(data: Data(json.utf8))
        XCTAssertEqual(tokenizer.decode(ids: [0, 4, 7, 2]), "the to")
    }

    /// The official microsoft/trocr-*-printed checkpoints happen to start
    /// generation from eos_token_id (a documented quirk) -- but that's a
    /// per-checkpoint fact, not a universal rule: a fine-tune can (and one
    /// does) start from bos instead. decoderStartId must reflect whatever
    /// tokenizer.json actually says, not assume it equals eosId.
    func testDecoderStartIdReflectsExportedValueNotAssumedEqualToEos() throws {
        let quirky = try TrOCRTokenizer(data: Data("""
        {"kind":"bpe","bos_id":0,"eos_id":2,"pad_id":1,"unk_id":3,"decoder_start_id":2,"id_to_token":["<s>","<pad>","</s>","<unk>"]}
        """.utf8))
        XCTAssertEqual(quirky.decoderStartId, 2)

        let overridden = try TrOCRTokenizer(data: Data("""
        {"kind":"bpe","bos_id":0,"eos_id":2,"pad_id":1,"unk_id":3,"decoder_start_id":0,"id_to_token":["<s>","<pad>","</s>","<unk>"]}
        """.utf8))
        XCTAssertEqual(overridden.decoderStartId, 0)
    }
}

/// End-to-end smoke test for the CoreML pipeline (preprocessing -> encoder ->
/// greedy decode -> tokenizer). Does not assert exact text (that depends on
/// the checkpoint); pixel parity with PIL is pinned by TrOCRPreprocessTests and
/// accuracy is measured by training/eval/scripts/bench.py.
final class TrOCREndToEndTests: XCTestCase {
    static let repoRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()

    override class func setUp() {
        TrOCREngine.resourceRootOverride = repoRoot.appendingPathComponent("Vendor/trocr")
    }

    func testEachModelSizeProducesNonEmptyText() async throws {
        let png = Self.repoRoot.appendingPathComponent("Tests/Fixtures/ocr/1.png")
        let image = try XCTUnwrap(loadCGImage(png))

        for size in TrOCRModelSize.allCases {
            let recognized = try await TrOCREngine.recognizeText(in: image, model: size)
            XCTAssertFalse(recognized.trimmingCharacters(in: .whitespaces).isEmpty, "TrOCR \(size.rawValue) returned empty text")
        }
    }

    private func loadCGImage(_ url: URL) -> CGImage? {
        NSImage(contentsOf: url)?.cgImage(forProposedRect: nil, context: nil, hints: nil)
    }
}

final class TrOCRLineSplitTests: XCTestCase {
    private static let monorepoRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()   // Tests
        .deletingLastPathComponent()   // apps/mac
        .deletingLastPathComponent()   // apps
        .deletingLastPathComponent()   // repo root

    /// These frozen eval fixtures are each ONE text line whose above-base Khmer
    /// marks form a separate ink run 37-46% of the base line's height; they
    /// must not be read as extra lines. ui-008 is dark mode, ui-005 light.
    func testSingleLineFixturesWithDetachedMarksStayOneStrip() throws {
        for name in ["ui/ui-008.png", "ui/ui-005.png", "ui/ui-009.png", "printed/printed-024.png"] {
            let url = Self.monorepoRoot.appendingPathComponent("training/eval/fixtures/\(name)")
            let image = try XCTUnwrap(NSImage(contentsOf: url)?.cgImage(forProposedRect: nil, context: nil, hints: nil), name)
            XCTAssertEqual(TrOCREngine.lineStrips(image).count, 1, name)
        }
    }

    /// TrOCR decodes one line per pass, so lineStrips must find each text line
    /// (including Khmer with marks above/below) or the output loses its
    /// linebreaks.
    func testLineStripsSplitsMultiLineClipIntoOneStripPerLine() throws {
        let lines = ["ច្បាប់ភូមិបាល", "ផ្តល់សិទ្ធិអំណាចឲ្យរដ្ឋ", "Hello world"]
        let size = NSSize(width: 600, height: 240)
        let rep = try XCTUnwrap(NSBitmapImageRep(
            bitmapDataPlanes: nil, pixelsWide: Int(size.width), pixelsHigh: Int(size.height),
            bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
            colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0))
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
        NSColor.white.setFill()
        NSRect(origin: .zero, size: size).fill()
        let attrs: [NSAttributedString.Key: Any] = [.font: NSFont.systemFont(ofSize: 36), .foregroundColor: NSColor.black]
        for (i, line) in lines.enumerated() {
            (line as NSString).draw(at: NSPoint(x: 20, y: size.height - CGFloat(i + 1) * 75), withAttributes: attrs)
        }
        NSGraphicsContext.restoreGraphicsState()
        let image = try XCTUnwrap(rep.cgImage)

        XCTAssertEqual(TrOCREngine.lineStrips(image).count, 3)
    }

    /// At normal Khmer line heights the subscripts of one line touch the
    /// above-marks of the next, so no blank row separates them. Four such
    /// lines (mirrors training/eval/tests/test_pipeline.py touching_lines)
    /// must still come out as four strips covering the whole image.
    func testLineStripsSplitsTouchingLinesAtThePitch() throws {
        let width = 260, pitch = 22, n = 4, height = n * pitch + 4
        let ctx = try XCTUnwrap(CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                                          bytesPerRow: 0, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                          bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        ctx.setFillColor(.white)
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        ctx.setFillColor(.black)
        // PIL-style inclusive (x0, y0, x1, y1) boxes, y down from the top.
        func box(_ x0: Int, _ y0: Int, _ x1: Int, _ y1: Int) {
            ctx.fill(CGRect(x: x0, y: height - 1 - y1, width: x1 - x0 + 1, height: y1 - y0 + 1))
        }
        for i in 0..<n {
            let top = 2 + i * pitch
            box(10, top + 7, width - 10, top + 15)          // dense core band
            box(30, top, 60, top + 6)                       // above marks
            box(90, top + 15, 120, top + pitch + 1)         // subscripts
            box(150, top + 15, 151, top + pitch + 8)        // bridge into the next line
        }
        let image = try XCTUnwrap(ctx.makeImage())

        let strips = TrOCREngine.lineStrips(image)
        XCTAssertEqual(strips.count, 4)
        XCTAssertEqual(strips.map(\.height).reduce(0, +), height)
    }
}

final class TrOCRPreprocessTests: XCTestCase {
    private static let macRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent()

    /// Training resizes with PIL bilinear (OCRDataset._preprocess_image);
    /// on-device pixels must match it or greedy decode drifts.
    func testResizedRGBMatchesPILBilinear() throws {
        let png = Self.macRoot.deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("training/eval/fixtures/printed/printed-001.png")
        let image = try XCTUnwrap(NSImage(contentsOf: png)?.cgImage(forProposedRect: nil, context: nil, hints: nil))
        let golden = try [UInt8](Data(contentsOf: Self.macRoot.appendingPathComponent("Tests/Fixtures/ocr/printed-001.pil384.rgb")))

        let swift = try XCTUnwrap(TrOCREngine.resizedRGB(image, size: 384))

        XCTAssertEqual(swift.count, golden.count)
        let maxDiff = zip(swift, golden).map { abs(Int($0) - Int($1)) }.max() ?? 0
        XCTAssertLessThanOrEqual(maxDiff, 1)
    }
}

final class TrOCRDecodeGuardTests: XCTestCase {
    func testBlockedTokenCapsRuns() {
        XCTAssertNil(TrOCREngine.blockedToken([0, 5, 5, 5]))           // 3 in a row: allowed
        XCTAssertEqual(TrOCREngine.blockedToken([0, 5, 5, 5, 5]), 5)   // 4 in a row: block a 5th
        XCTAssertNil(TrOCREngine.blockedToken([0, 5, 6, 5, 6]))        // alternating: not a run
        XCTAssertNil(TrOCREngine.blockedToken([0]))
    }

    func testArgmaxSkipsBlockedToken() throws {
        let logits = try MLMultiArray(shape: [1, 1, 4], dataType: .float32)
        for (i, v) in [0.1, 0.9, 0.5, 0.2].enumerated() { logits[i] = NSNumber(value: v) }
        XCTAssertEqual(TrOCREngine.argmaxLastStep(logits), 1)
        XCTAssertEqual(TrOCREngine.argmaxLastStep(logits, excluding: 1), 2)
    }

    func testContainsKhmerRoutesLatinOnlyLines() {
        XCTAssertFalse(TrOCREngine.containsKhmer("Warranty 11 months"))
        XCTAssertFalse(TrOCREngine.containsKhmer(""))
        XCTAssertTrue(TrOCREngine.containsKhmer("កុំព្យូទ័រ & គ្រឿងបន្លាស់"))
        XCTAssertTrue(TrOCREngine.containsKhmer("ដក language"))
    }

    func testTesseractTSVGivesWordsAndLowestConfidence() {
        let tsv = """
        level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext
        1\t1\t0\t0\t0\t0\t0\t0\t410\t72\t-1\t
        5\t1\t1\t1\t1\t1\t30\t10\t120\t40\t95.0\tWarranty
        5\t1\t1\t1\t1\t2\t160\t10\t30\t40\t72.5\t11
        """
        let r = TesseractEngine.parseTSV(tsv)
        XCTAssertEqual(r.text, "Warranty 11")
        XCTAssertEqual(r.minConfidence, 72.5)
        XCTAssertEqual(TesseractEngine.parseTSV("").minConfidence, 0)
    }

    func testDecodeCapMatchesTrainingMaxLabelLength() {
        XCTAssertEqual(TrOCREngine.maxLength, 128)   // training MAX_LABEL_LEN
    }
}

extension TrOCREndToEndTests {
    /// printed-080 is a 75-code-point line (~1 token per Khmer code point); the
    /// old 64-token cap cut it off. Count unicode scalars, not Characters:
    /// a Khmer cluster (consonant + coeng + subscript + vowel) is ONE Character.
    func testLongLineIsNotTruncatedAt64Tokens() async throws {
        let png = Self.repoRoot.deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("training/eval/fixtures/printed/printed-080.png")
        let image = try XCTUnwrap(NSImage(contentsOf: png)?.cgImage(forProposedRect: nil, context: nil, hints: nil))
        let text = try await TrOCREngine.recognizeText(in: image, model: .khmer)
        XCTAssertGreaterThan(text.unicodeScalars.count, 64)
    }

    /// Each line gets a fresh, zeroed KV state: recognizing A, then B, then A
    /// again must give the same A (no stale cache leaking between calls).
    func testRepeatedRecognitionIsDeterministic() async throws {
        let fixtures = Self.repoRoot.deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("training/eval/fixtures")
        func load(_ name: String) throws -> CGImage {
            try XCTUnwrap(NSImage(contentsOf: fixtures.appendingPathComponent(name))?
                .cgImage(forProposedRect: nil, context: nil, hints: nil))
        }
        let a = try load("printed/printed-001.png"), b = try load("ui/ui-008.png")
        let first = try await TrOCREngine.recognizeText(in: a, model: .khmer)
        _ = try await TrOCREngine.recognizeText(in: b, model: .khmer)
        let again = try await TrOCREngine.recognizeText(in: a, model: .khmer)
        XCTAssertEqual(first, again)
        XCTAssertFalse(first.isEmpty)
    }
}
