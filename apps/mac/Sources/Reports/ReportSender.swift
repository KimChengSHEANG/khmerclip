import Foundation

/// Shared enqueue -> flush -> message flow for capture reports and feedback.
enum ReportSender {
    static let saveFailedMessage = "Couldn't save the report. Please try again."
    static let sentMessage = "Thanks — your report was sent."
    static let queuedMessage = "Report saved. It will be sent when you're online."

    /// Queues `payload` (throws if it can't be saved -- nothing is sent then), then flushes in the
    /// background and calls `done` on the main actor with the message to show.
    @MainActor
    static func queueAndSend(_ payload: [String: Any], done: @escaping @MainActor (String) -> Void) throws {
        try ReportOutbox.shared.enqueue(payload)
        Task {
            let sent = await ReportOutbox.shared.flush(post: { await ReportOutbox.post($0, to: ReportOutbox.reportURL) })
            done(sent >= 1 ? sentMessage : queuedMessage)
        }
    }
}
