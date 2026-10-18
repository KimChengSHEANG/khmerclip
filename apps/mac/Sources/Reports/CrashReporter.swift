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
        guard let first = text.split(separator: "\n", omittingEmptySubsequences: true).first else { return }
        _ = try? outbox.enqueue(Report.crash(signature: String(first), details: text, engine: engine))
    }
}
