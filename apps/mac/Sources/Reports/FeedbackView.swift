import SwiftUI

/// Free-form feedback form. Calls `onSend(message, email)` with trimmed values; the host decides
/// whether to close the window (it stays open, text intact, if the report couldn't be saved).
struct FeedbackView: View {
    let onSend: (String, String) -> Void
    @State private var message = ""
    @State private var email = ""

    static func canSend(_ message: String) -> Bool {
        !message.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Tell us what's on your mind — a bug, an idea, or anything else.")
            TextEditor(text: $message)
                .font(.body)
                .frame(minHeight: 140)
                .overlay(RoundedRectangle(cornerRadius: 4).stroke(Color.secondary.opacity(0.4)))
            TextField("Your email (if you'd like a reply)", text: $email)
                .textFieldStyle(.roundedBorder)
            HStack {
                Spacer()
                Button("Send") {
                    onSend(message.trimmingCharacters(in: .whitespacesAndNewlines),
                           email.trimmingCharacters(in: .whitespacesAndNewlines))
                }
                .keyboardShortcut(.defaultAction)
                .disabled(!Self.canSend(message))
            }
        }
        .padding(16)
        .frame(width: 420)
    }
}
