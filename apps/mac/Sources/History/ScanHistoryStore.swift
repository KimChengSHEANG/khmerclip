import AppKit
import Combine

/// Scan history, plus (opt-in) each capture's image in `captures/<id>.png` with a
/// `<id>.json` of OCR details. "Flag as Wrong" copies that pair into `feedback/`
/// for the eval set (training/eval/scripts/draft_real_fixtures.py). Same layout
/// as the desktop app's HistoryStore.
final class ScanHistoryStore: ObservableObject {
    @Published private(set) var records: [ScanRecord] = []
    /// Bumped on flag/purge so the history list re-reads capture state.
    @Published private(set) var capturesVersion = 0

    private let maxRecords = 200
    private let fileURL: URL

    static let dataDir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("KhmerClip", isDirectory: true)
    static let capturesDir = dataDir.appendingPathComponent("captures", isDirectory: true)
    static let feedbackDir = dataDir.appendingPathComponent("feedback", isDirectory: true)

    init() {
        try? FileManager.default.createDirectory(at: Self.dataDir, withIntermediateDirectories: true)
        fileURL = Self.dataDir.appendingPathComponent("history.json")
        load()
    }

    /// `image` is kept with `meta` only when given -- the caller checks the preference.
    func add(_ record: ScanRecord, image: CGImage? = nil, meta: [String: Any] = [:]) {
        if let image { saveCapture(id: record.id, image: image, meta: meta) }
        records.insert(record, at: 0)
        if records.count > maxRecords {
            records[maxRecords...].forEach { deleteCapture($0.id) }
            records.removeLast(records.count - maxRecords)
        }
        save()
    }

    func clear() {
        records.removeAll()
        Self.purgeCaptures()
        save()
    }

    func remove(_ id: UUID) {
        records.removeAll { $0.id == id }
        deleteCapture(id)
        save()
    }

    func hasCapture(_ id: UUID) -> Bool {
        FileManager.default.fileExists(atPath: Self.file(Self.capturesDir, id, "png").path)
    }

    func isFlagged(_ id: UUID) -> Bool {
        FileManager.default.fileExists(atPath: Self.file(Self.feedbackDir, id, "png").path)
    }

    /// Copies the capture pair into feedback/. Flagged clips outlive the history.
    func flag(_ id: UUID) {
        let fm = FileManager.default
        try? fm.createDirectory(at: Self.feedbackDir, withIntermediateDirectories: true)
        for ext in ["png", "json"] {
            let dest = Self.file(Self.feedbackDir, id, ext)
            try? fm.removeItem(at: dest)
            try? fm.copyItem(at: Self.file(Self.capturesDir, id, ext), to: dest)
        }
        capturesVersion += 1
    }

    func purgeCaptures() {
        Self.purgeCaptures()
        capturesVersion += 1
    }

    private static func purgeCaptures() {
        try? FileManager.default.removeItem(at: capturesDir)
    }

    private static func file(_ dir: URL, _ id: UUID, _ ext: String) -> URL {
        dir.appendingPathComponent("\(id.uuidString).\(ext)")
    }

    private func saveCapture(id: UUID, image: CGImage, meta: [String: Any]) {
        try? FileManager.default.createDirectory(at: Self.capturesDir, withIntermediateDirectories: true)
        guard let png = NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:]),
              let json = try? JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .sortedKeys])
        else { return }
        try? png.write(to: Self.file(Self.capturesDir, id, "png"))
        try? json.write(to: Self.file(Self.capturesDir, id, "json"))
    }

    private func deleteCapture(_ id: UUID) {
        for ext in ["png", "json"] {
            try? FileManager.default.removeItem(at: Self.file(Self.capturesDir, id, ext))
        }
    }

    private func load() {
        guard let data = try? Data(contentsOf: fileURL) else { return }
        records = (try? JSONDecoder().decode([ScanRecord].self, from: data)) ?? []
    }

    private func save() {
        guard let data = try? JSONEncoder().encode(records) else { return }
        try? data.write(to: fileURL, options: .atomic)
    }
}
