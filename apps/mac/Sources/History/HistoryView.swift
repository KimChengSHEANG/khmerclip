import SwiftUI

struct HistoryView: View {
    @ObservedObject var store: ScanHistoryStore
    @State private var reporting: ScanRecord?
    @State private var alertMessage: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Scan History").font(.headline)
                Spacer()
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
                        HStack(spacing: 10) {
                            let hasCapture = store.hasCapture(record.id)
                            if hasCapture, let image = NSImage(contentsOf: store.captureURL(record.id)) {
                                Image(nsImage: image)
                                    .resizable()
                                    .scaledToFit()
                                    .frame(width: 72, height: 40)
                            }
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
                            if hasCapture {
                                if store.isFlagged(record.id) {
                                    Button("Reported") {}.disabled(true)
                                } else {
                                    Button("Report") { reporting = record }
                                }
                            }
                        }
                        .padding(.vertical, 4)
                        .contextMenu {
                            Button("Copy") { ClipboardManager.copy(record.text) }
                            Button("Speak") { TextToSpeechManager.shared.speak(record.text) }
                            Button("Translate") { TranslateWindowPresenter.show(text: record.text) }
                            Divider()
                            Button("Delete", role: .destructive) { store.remove(record.id) }
                        }
                    }
                }
            }
        }
        .frame(width: 520, height: 480)
        .sheet(item: $reporting) { record in
            ReportView(imageURL: store.captureURL(record.id), ocrText: ocrText(for: record),
                       onSend: { send(record, correction: $0) }, onCancel: { reporting = nil })
        }
        .alert(alertMessage ?? "", isPresented: Binding(get: { alertMessage != nil },
                                                        set: { if !$0 { alertMessage = nil } })) {
            Button("OK") {}
        }
    }

    private func ocrText(for record: ScanRecord) -> String {
        (store.captureMeta(record.id)["ocr_text"] as? String) ?? record.text
    }

    /// Queue first; only a saved report marks the row Reported.
    private func send(_ record: ScanRecord, correction: String) {
        reporting = nil
        let payload = Report.capture(pngURL: store.captureURL(record.id), meta: store.captureMeta(record.id),
                                     ocrText: ocrText(for: record), correction: correction,
                                     engine: UserDefaults.standard.string(forKey: "ocrEngine") ?? "tesseract")
        do {
            try ReportSender.queueAndSend(payload) { alertMessage = $0 }
            store.flag(record.id, correction: correction)
        } catch {
            alertMessage = ReportSender.saveFailedMessage
        }
    }
}
