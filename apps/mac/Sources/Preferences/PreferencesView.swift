import SwiftUI

struct PreferencesView: View {
    @EnvironmentObject var preferences: AppPreferences

    var body: some View {
        Form {
            Section("Shortcuts") {
                LabeledContent("Capture Text") {
                    HotKeyRecorderView(definition: $preferences.captureHotKey)
                        .frame(width: 140, height: 24)
                }
                LabeledContent("Scan QR / Barcode") {
                    HotKeyRecorderView(definition: $preferences.qrHotKey)
                        .frame(width: 140, height: 24)
                }
            }

            Section("Behavior") {
                Toggle("Auto-copy result to clipboard", isOn: $preferences.autoCopyToClipboard)
                Toggle("Show result popup", isOn: $preferences.showResultHUD)
                Toggle("Play sound on capture", isOn: $preferences.playCaptureSound)
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
