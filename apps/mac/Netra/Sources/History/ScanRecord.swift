import Foundation

struct ScanRecord: Codable, Identifiable, Equatable {
    let id: UUID
    let text: String
    let date: Date
    let kind: Kind

    enum Kind: String, Codable {
        case text
        case barcode
    }

    init(text: String, kind: Kind = .text, date: Date = Date()) {
        self.id = UUID()
        self.text = text
        self.kind = kind
        self.date = date
    }
}
