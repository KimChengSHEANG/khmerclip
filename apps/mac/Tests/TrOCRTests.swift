import XCTest
import AppKit
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
/// greedy decode -> tokenizer). Deliberately does NOT assert exact text:
/// measured against the reference PyTorch/HF pipeline, on-device pixel
/// values differ by up to ~0.5 (of a [-1,1] range) because CGContext's
/// resampler is a different implementation than PIL's -- not a bug, but a
/// real, unresolved precision gap between this Swift preprocessing and the
/// Python reference used when the models were converted. That gap is enough
/// to change greedy-decoded words on some inputs. Fixing it would mean
/// reimplementing PIL's resampling filters in Swift, which is out of scope
/// here. This test only proves the pipeline runs and produces text.
final class TrOCREndToEndTests: XCTestCase {
    private static let repoRoot = URL(fileURLWithPath: #filePath)
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
}
