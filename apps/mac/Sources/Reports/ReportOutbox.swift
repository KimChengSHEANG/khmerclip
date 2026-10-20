import Foundation

/// Queues report payloads as JSON files and sends them to the relay. Failures never throw
/// into the app; unsent reports stay queued and are retried on the next flush.
struct ReportOutbox {
    let dir: URL

    static let unsetURL = URL(string: "https://khmerclipper-report-relay.REPLACE_ME.workers.dev/report")!
    static var reportURL = unsetURL
    static let appKey = "kc-reports-1"
    /// A crash loop must not flood the outbox (or the relay).
    static let maxQueuedCrashes = 20

    /// 200 = sent; 400/401/413 = rejected for good. Both are deleted, anything else is retried.
    private static let doneStatuses: Set<Int> = [200, 400, 401, 413]
    private static let flushGate = FlushGate()

    @discardableResult
    func enqueue(_ payload: [String: Any]) throws -> URL {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let id = payload["id"] as? String ?? UUID().uuidString.lowercased()
        let url = dir.appendingPathComponent("\(id).json")
        let data = try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
        try data.write(to: url, options: .atomic)
        return url
    }

    /// Queues a crash report unless one with the same signature is already queued or the outbox
    /// already holds `maxQueuedCrashes` crashes. Returns nil when skipped.
    @discardableResult
    func enqueueCrash(_ payload: [String: Any]) throws -> URL? {
        let crashes = queued().compactMap(Self.load).filter { $0["type"] as? String == "crash" }
        let signature = payload["signature"] as? String
        if crashes.count >= Self.maxQueuedCrashes || crashes.contains(where: { $0["signature"] as? String == signature }) {
            return nil
        }
        return try enqueue(payload)
    }

    private func queued() -> [URL] {
        ((try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension == "json" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    private static func load(_ url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }

    func dropType(_ type: String) {
        for url in queued() where Self.load(url)?["type"] as? String == type {
            try? FileManager.default.removeItem(at: url)
        }
    }

    /// Sends every queued report; returns how many the relay accepted. Only one flush runs
    /// at a time: a call that overlaps a running flush returns 0 without sending.
    func flush(url: URL = ReportOutbox.reportURL, post: ([String: Any]) async -> Int) async -> Int {
        if url.absoluteString.contains("REPLACE_ME") { return 0 }
        guard Self.flushGate.begin() else { return 0 }
        defer { Self.flushGate.end() }
        var sent = 0
        for file in queued() {
            guard let payload = Self.load(file) else {  // unreadable: can never be sent
                try? FileManager.default.removeItem(at: file)
                continue
            }
            let status = await post(payload)
            if status == 200 { sent += 1 }
            if Self.doneStatuses.contains(status) { try? FileManager.default.removeItem(at: file) }
        }
        return sent
    }

    /// POSTs one payload; returns the HTTP status, or 0 on any failure.
    static func post(_ payload: [String: Any], to url: URL) async -> Int {
        do {
            var request = URLRequest(url: url, timeoutInterval: 20)
            request.httpMethod = "POST"
            request.httpBody = try JSONSerialization.data(withJSONObject: payload)
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.setValue(appKey, forHTTPHeaderField: "X-KhmerClipper-Key")
            let version = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "dev"
            request.setValue("KhmerClipper/\(version)", forHTTPHeaderField: "User-Agent")
            let (_, response) = try await URLSession.shared.data(for: request)
            return (response as? HTTPURLResponse)?.statusCode ?? 0
        } catch {
            return 0
        }
    }
}

private final class FlushGate: @unchecked Sendable {
    private let lock = NSLock()
    private var running = false

    func begin() -> Bool {
        lock.lock(); defer { lock.unlock() }
        if running { return false }
        running = true
        return true
    }

    func end() {
        lock.lock(); running = false; lock.unlock()
    }
}
