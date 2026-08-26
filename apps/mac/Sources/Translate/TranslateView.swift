import SwiftUI
import Translation
import AppKit

/// Presented as its own window when the user picks "Translate" from the result
/// HUD or history list. Uses Apple's on-device Translation framework.
struct TranslateView: View {
    let sourceText: String

    @State private var translatedText: String = ""
    @State private var isTranslating = true
    @State private var configuration: TranslationSession.Configuration?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Original").font(.caption).foregroundStyle(.secondary)
            ScrollView { Text(sourceText).textSelection(.enabled) }
                .frame(maxHeight: 120)

            Divider()

            Text("Translation").font(.caption).foregroundStyle(.secondary)
            if isTranslating {
                ProgressView().controlSize(.small)
            } else {
                ScrollView { Text(translatedText).textSelection(.enabled) }
                    .frame(maxHeight: 120)
            }

            HStack {
                Spacer()
                Button("Copy Translation") {
                    ClipboardManager.copy(translatedText)
                }
                .disabled(translatedText.isEmpty)
            }
        }
        .padding()
        .frame(width: 380)
        .translationTask(configuration) { session in
            do {
                let response = try await session.translate(sourceText)
                translatedText = response.targetText
            } catch {
                translatedText = "Translation failed: \(error.localizedDescription)"
            }
            isTranslating = false
        }
        .onAppear {
            configuration = TranslationSession.Configuration(source: nil, target: nil)
        }
    }
}
