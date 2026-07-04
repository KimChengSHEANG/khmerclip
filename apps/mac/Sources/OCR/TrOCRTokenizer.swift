import Foundation

/// Decodes token ids produced by the TrOCR decoder back into text. Only
/// decode is implemented (id -> text) since the encoder side never tokenizes
/// text -- TrOCR's only text input is the image.
///
/// Two tokenizer families appear across `microsoft/trocr-*-printed`:
/// - `bpe`: GPT2/RoBERTa byte-level BPE (trocr-base, trocr-large). Pieces use
///   'Ġ' for a leading space and byte-map non-ASCII bytes to printable
///   unicode scalars; decoding must reverse that byte mapping.
/// - `sentencepiece`: XLM-RoBERTa SentencePiece (trocr-small). Pieces are
///   literal UTF-8 text with '▁' marking a leading space.
///
/// The khmer fine-tune is `bpe` but with a twist: it appends literal Khmer
/// tokens (raw UTF-8, not byte-mapped) past the base vocab. Those ids are
/// listed in `literal_ids` and decoded verbatim -- see decode()'s bpe branch.
struct TrOCRTokenizer {
    enum Kind: String, Decodable {
        case bpe
        case sentencepiece
    }

    private struct Payload: Decodable {
        let kind: Kind
        let bosId: Int
        let eosId: Int
        let padId: Int
        let unkId: Int
        let decoderStartId: Int
        let idToToken: [String]
        let literalIds: [Int]?

        enum CodingKeys: String, CodingKey {
            case kind, bosId = "bos_id", eosId = "eos_id", padId = "pad_id", unkId = "unk_id"
            case decoderStartId = "decoder_start_id", idToToken = "id_to_token", literalIds = "literal_ids"
        }
    }

    let kind: Kind
    let eosId: Int
    /// Where greedy decode starts from. NOT always eos_token_id: true for the
    /// official microsoft/trocr-*-printed checkpoints (a documented quirk),
    /// but fine-tunes can set it independently -- e.g. a Khmer fine-tune
    /// starting from bos instead. Exported per-model rather than assumed.
    let decoderStartId: Int
    private let idToToken: [String]
    private let specialIds: Set<Int>
    /// Added-vocab ids whose piece is literal UTF-8 text (e.g. Khmer glyphs on
    /// the khmer fine-tune) rather than GPT-2 byte-BPE. Empty for the English
    /// checkpoints. See decode()'s bpe branch.
    private let literalIds: Set<Int>

    init(data: Data) throws {
        let payload = try JSONDecoder().decode(Payload.self, from: data)
        kind = payload.kind
        eosId = payload.eosId
        decoderStartId = payload.decoderStartId
        idToToken = payload.idToToken
        specialIds = [payload.bosId, payload.eosId, payload.padId, payload.unkId]
        literalIds = Set(payload.literalIds ?? [])
    }

    func decode(ids: [Int]) -> String {
        let kept = ids.filter { !specialIds.contains($0) && $0 >= 0 && $0 < idToToken.count }

        switch kind {
        case .sentencepiece:
            let joined = kept.map { idToToken[$0] }.joined().replacingOccurrences(of: "\u{2581}", with: " ")
            return joined.trimmingCharacters(in: .whitespaces)
        case .bpe:
            // The khmer fine-tune's vocab mixes GPT-2 byte-BPE pieces (whose
            // scalars byte-map back to UTF-8 bytes) with literal added tokens
            // (raw Khmer text). Mirror HF: accumulate byte-BPE pieces into a
            // buffer, and when a literal token appears flush the buffer to text
            // and append the token verbatim. Pure-English models have no literal
            // ids, so this reduces to a single byte-decode.
            var result = ""
            var bytes: [UInt8] = []
            func flush() {
                if !bytes.isEmpty { result += String(decoding: bytes, as: UTF8.self); bytes.removeAll(keepingCapacity: true) }
            }
            for id in kept {
                let token = idToToken[id]
                if literalIds.contains(id) {
                    flush()
                    result += token
                } else {
                    for scalar in token.unicodeScalars {
                        if let byte = Self.byteDecoder[scalar] { bytes.append(byte) }
                    }
                }
            }
            flush()
            return result
        }
    }

    /// GPT-2's byte<->unicode mapping (see openai/gpt-2 `bytes_to_unicode`):
    /// printable bytes map to themselves, the rest map to unicode scalars
    /// starting at 256 so every byte value round-trips through a valid,
    /// visible unicode string.
    private static let byteDecoder: [Unicode.Scalar: UInt8] = {
        let printable = Set(33...126).union(161...172).union(174...255)
        var encoder: [Int: UInt8] = [:]
        for b in printable { encoder[b] = UInt8(b) }
        var n = 0
        for b in 0..<256 where !printable.contains(b) {
            encoder[256 + n] = UInt8(b)
            n += 1
        }
        var decoder: [Unicode.Scalar: UInt8] = [:]
        for (scalarValue, byte) in encoder {
            decoder[Unicode.Scalar(scalarValue)!] = byte
        }
        return decoder
    }()
}
