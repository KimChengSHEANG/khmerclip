import SwiftUI

struct HistoryView: View {
    @ObservedObject var store: ScanHistoryStore

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Scan History").font(.headline)
                Spacer()
                if FileManager.default.fileExists(atPath: ScanHistoryStore.feedbackDir.path) {
                    Button("Feedback Folder") { NSWorkspace.shared.open(ScanHistoryStore.feedbackDir) }
                }
                Button("Clear") { store.clear() }
                    .disabled(store.records.isEmpty)
            }
            .padding()

            Divider()

            if store.records.isEmpty {
                Spacer()
                Text("No scans yet").foregroundStyle(.secondary)
                Spacer()
            } else {
                List {
                    ForEach(store.records) { record in
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Image(systemName: record.kind == .barcode ? "qrcode" : "text.viewfinder")
                                Text(record.date, style: .relative)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                Spacer()
                            }
                            Text(record.text)
                                .lineLimit(3)
                                .textSelection(.enabled)
                        }
                        .padding(.vertical, 4)
                        .contextMenu {
                            Button("Copy") { ClipboardManager.copy(record.text) }
                            Button("Speak") { TextToSpeechManager.shared.speak(record.text) }
                            Button("Translate") { TranslateWindowPresenter.show(text: record.text) }
                            if store.hasCapture(record.id) {
                                if store.isFlagged(record.id) {
                                    Button("Flagged") {}.disabled(true)
                                } else {
                                    Button("Flag as Wrong") { store.flag(record.id) }
                                }
                            }
                            Divider()
                            Button("Delete", role: .destructive) { store.remove(record.id) }
                        }
                    }
                }
            }
        }
        .frame(width: 420, height: 480)
    }
}
