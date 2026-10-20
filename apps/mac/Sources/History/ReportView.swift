import SwiftUI

struct ReportView: View {
    let imageURL: URL
    let ocrText: String
    let onSend: (String) -> Void
    let onCancel: () -> Void

    @State private var correction: String

    init(imageURL: URL, ocrText: String, onSend: @escaping (String) -> Void, onCancel: @escaping () -> Void) {
        self.imageURL = imageURL
        self.ocrText = ocrText
        self.onSend = onSend
        self.onCancel = onCancel
        _correction = State(initialValue: ocrText)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if let image = NSImage(contentsOf: imageURL) {
                Image(nsImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: 480)
            }
            Text("KhmerClipper read").font(.headline)
            Text(ocrText)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
            Text("What it should say").font(.headline)
            TextEditor(text: $correction)
                .frame(height: 80)
                .border(Color.secondary.opacity(0.4))
            Text("This image and text will be sent to the KhmerClipper developer.")
                .font(.caption)
                .foregroundStyle(.secondary)
            HStack {
                Spacer()
                Button("Cancel", action: onCancel).keyboardShortcut(.cancelAction)
                Button("Send") {
                    let edited = correction.trimmingCharacters(in: .whitespacesAndNewlines)
                    onSend(edited == ocrText.trimmingCharacters(in: .whitespacesAndNewlines) ? "" : correction)
                }
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding()
        .frame(width: 512)
    }
}
