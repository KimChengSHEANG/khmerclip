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
        {"kind":"bpe","bos_id":0,"eos_id":2,"pad_id":1,"unk_id":3,
         "id_to_token":["<s>","<pad>","</s>","<unk>",".","Ġthe",",","Ġto"]}
        """
        let tokenizer = try TrOCRTokenizer(data: Data(json.utf8))
        XCTAssertEqual(tokenizer.decode(ids: [0, 5, 7, 2]), " the to")
    }

    func testSentencePieceDecodeReplacesMarkerAndTrimsLeadingSpace() throws {
        let json = """
        {"kind":"sentencepiece","bos_id":0,"eos_id":2,"pad_id":1,"unk_id":3,
         "id_to_token":["<s>","<pad>","</s>","<unk>","▁the",".",",","▁to"]}
        """
        let tokenizer = try TrOCRTokenizer(data: Data(json.utf8))
        XCTAssertEqual(tokenizer.decode(ids: [0, 4, 7, 2]), "the to")
    }

    func testDecoderStartIdEqualsEosId() throws {
        let json = """
        {"kind":"bpe","bos_id":0,"eos_id":2,"pad_id":1,"unk_id":3,"id_to_token":["<s>","<pad>","</s>","<unk>"]}
        """
        let tokenizer = try TrOCRTokenizer(data: Data(json.utf8))
        XCTAssertEqual(tokenizer.decoderStartId, tokenizer.eosId)
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
