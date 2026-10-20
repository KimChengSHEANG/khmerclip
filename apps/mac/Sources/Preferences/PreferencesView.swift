import SwiftUI

struct PreferencesView: View {
    @EnvironmentObject var preferences: AppPreferences

    var body: some View {
        Form {
            Section("Shortcuts") {
                LabeledContent("Capture Text") {
                    HotKeyRecorderView(defaultsKey: AppPreferences.ShortcutKeys.capture)
                        .frame(width: 140, height: 24)
                }
            }

            Section("OCR Engine") {
                Picker("Engine", selection: $preferences.ocrEngine) {
                    ForEach(OCREngineKind.allCases) { engine in
                        Text(engine.displayName).tag(engine)
                    }
                }
            }

            Section("Behavior") {
                Toggle("Auto-copy result to clipboard", isOn: $preferences.autoCopyToClipboard)
                Toggle("Preserve linebreaks", isOn: $preferences.preserveLinebreaks)
                Toggle("Show result popup", isOn: $preferences.showResultHUD)
                Toggle("Play sound on capture", isOn: $preferences.playCaptureSound)
                Toggle("Keep capture images for accuracy feedback", isOn: $preferences.keepCaptureImages)
                Toggle("Send crash reports automatically", isOn: $preferences.sendCrashReports)
                Toggle("Launch at login", isOn: Binding(
                    get: { preferences.launchAtLogin },
                    set: { newValue in
                        preferences.launchAtLogin = newValue
                        LaunchAtLoginManager.setEnabled(newValue)
                    }
                ))
            }
        }
        .padding(20)
        .frame(width: 420)
    }
}
