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

        enum CodingKeys: String, CodingKey {
            case kind, bosId = "bos_id", eosId = "eos_id", padId = "pad_id", unkId = "unk_id"
            case decoderStartId = "decoder_start_id", idToToken = "id_to_token"
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

    init(data: Data) throws {
        let payload = try JSONDecoder().decode(Payload.self, from: data)
        kind = payload.kind
        eosId = payload.eosId
        decoderStartId = payload.decoderStartId
        idToToken = payload.idToToken
        specialIds = [payload.bosId, payload.eosId, payload.padId, payload.unkId]
    }

    func decode(ids: [Int]) -> String {
        let tokens = ids
            .filter { !specialIds.contains($0) }
            .compactMap { $0 >= 0 && $0 < idToToken.count ? idToToken[$0] : nil }

        switch kind {
        case .sentencepiece:
            let joined = tokens.joined().replacingOccurrences(of: "\u{2581}", with: " ")
            return joined.trimmingCharacters(in: .whitespaces)
        case .bpe:
            var bytes: [UInt8] = []
            bytes.reserveCapacity(tokens.reduce(0) { $0 + $1.count })
            for token in tokens {
                for scalar in token.unicodeScalars {
                    if let byte = Self.byteDecoder[scalar] {
                        bytes.append(byte)
                    }
                }
            }
            return String(decoding: bytes, as: UTF8.self)
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
