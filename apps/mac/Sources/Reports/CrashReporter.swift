import Darwin
import Foundation

/// Captures crashes to a pending file and turns it into a queued report on the next launch.
/// Swift-runtime traps that bypass the exception and signal handlers are an accepted ceiling.
enum CrashReporter {
    /// Path C string for the signal handler, prepared at install time and kept for the process lifetime.
    nonisolated(unsafe) fileprivate static var pendingPath: UnsafeMutablePointer<CChar>?
    nonisolated(unsafe) fileprivate static var pendingFile: URL?
    /// Set once the exception handler has written the pending file, so the SIGABRT that follows
    /// does not overwrite the exception's name, reason and stack.
    nonisolated(unsafe) fileprivate static var exceptionRecorded = false

    private static let signals: [Int32] = [SIGABRT, SIGSEGV, SIGBUS, SIGILL, SIGTRAP, SIGFPE]

    static func install(pendingURL: URL) {
        pendingFile = pendingURL
        pendingPath = strdup(pendingURL.path)

        NSSetUncaughtExceptionHandler { exception in
            let text = "\(exception.name.rawValue): \(exception.reason ?? "")\n"
                + exception.callStackSymbols.joined(separator: "\n")
            CrashReporter.exceptionRecorded = true
            if let url = CrashReporter.pendingFile { try? text.write(to: url, atomically: true, encoding: .utf8) }
        }

        // Alternate stack (kept for the process lifetime) so a stack-overflow SIGSEGV can still run the handler.
        let altSize = 64 * 1024
        var altStack = stack_t(ss_sp: malloc(altSize), ss_size: altSize, ss_flags: 0)
        sigaltstack(&altStack, nil)

        var action = sigaction()
        action.__sigaction_u.__sa_handler = { sig in
            // Async-signal-safe calls only: no allocation, no Swift/ObjC runtime work.
            guard !CrashReporter.exceptionRecorded, let path = CrashReporter.pendingPath else { signal(sig, SIG_DFL); raise(sig); return }
            let fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0o600)
            if fd >= 0 {
                withUnsafeTemporaryAllocation(of: UInt8.self, capacity: 32) { buf in
                    var n = 0
                    for c in "Signal ".utf8 { buf[n] = c; n += 1 }
                    // Signal numbers are below 100.
                    if sig >= 10 { buf[n] = UInt8(48 + sig / 10 % 10); n += 1 }
                    buf[n] = UInt8(48 + sig % 10); n += 1
                    buf[n] = 10; n += 1
                    _ = write(fd, buf.baseAddress, n)
                }
                withUnsafeTemporaryAllocation(of: UnsafeMutableRawPointer?.self, capacity: 64) { frames in
                    let depth = backtrace(frames.baseAddress, 64)
                    backtrace_symbols_fd(frames.baseAddress, depth, fd)
                }
                close(fd)
            }
            signal(sig, SIG_DFL)
            raise(sig)
        }
        action.sa_flags = SA_NODEFER | SA_RESETHAND | SA_ONSTACK
        sigemptyset(&action.sa_mask)
        for sig in signals { sigaction(sig, &action, nil) }
    }

    /// Queues the previous run's crash (if any and `enabled`), then always removes the pending file.
    /// Never throws: a failed enqueue just loses that one report.
    static func collect(pendingURL: URL, outbox: ReportOutbox, engine: String, enabled: Bool) {
        defer { try? FileManager.default.removeItem(at: pendingURL) }
        guard enabled else {
            outbox.dropType("crash")
            return
        }
        guard let data = try? Data(contentsOf: pendingURL), !data.isEmpty else { return }
        let text = String(decoding: data, as: UTF8.self)
        guard text.contains(where: { !$0.isWhitespace }) else { return }
        _ = try? outbox.enqueue(Report.crash(signature: signature(fromPending: text), details: text, engine: engine))
    }

    /// Fingerprint of a pending crash, stable across runs so the relay can dedup it:
    /// "Signal N at <symbol>" (first app frame outside this handler), or "<name>: <reason>"
    /// with addresses and numbers masked.
    static func signature(fromPending text: String, image: String = "KhmerClip") -> String {
        let lines = text.split(separator: "\n", omittingEmptySubsequences: true).map(String.init)
        guard let first = lines.first else { return "" }
        guard first.hasPrefix("Signal ") else {
            return first
                .replacingOccurrences(of: #"\b0x[0-9A-Fa-f]+\b"#, with: "<addr>", options: .regularExpression)
                .replacingOccurrences(of: #"\b[0-9]+\b"#, with: "<n>", options: .regularExpression)
        }
        for line in lines.dropFirst() {
            // backtrace_symbols format: "<index> <image> 0x<address> <symbol> + <offset>".
            let parts = line.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
            guard parts.count >= 4, parts[1] == image, parts[2].hasPrefix("0x") else { continue }
            let symbolParts = parts.count >= 6 && parts[parts.count - 2] == "+"
                ? parts[3..<(parts.count - 2)] : parts[3...]
            let symbol = symbolParts.joined(separator: " ")
            if symbol.contains("CrashReporter") { continue }  // the signal handler itself
            // No usable symbol (a stripped build): the address alone would not be stable.
            if symbol == image || symbol.hasPrefix("0x") || symbol == "__mh_execute_header" { return first }
            return "\(first) at \(symbol)"
        }
        return first
    }
}
