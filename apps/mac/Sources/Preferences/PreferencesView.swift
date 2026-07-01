import SwiftUI

struct PreferencesView: View {
    @EnvironmentObject var preferences: AppPreferences

    private let availableLanguages: [(code: String, name: String)] = [
        ("km-KH", "Khmer"), ("lo-LA", "Lao"), ("en-US", "English"), ("fr-FR", "French"),
        ("zh-Hans", "Chinese (Simplified)"), ("ja-JP", "Japanese"), ("ko-KR", "Korean"),
        ("th-TH", "Thai"), ("vi-VT", "Vietnamese"),
    ]

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

            Section("Recognition Languages") {
                ForEach(availableLanguages, id: \.code) { lang in
                    Toggle(lang.name, isOn: Binding(
                        get: { preferences.recognitionLanguages.contains(lang.code) },
                        set: { isOn in
                            if isOn {
                                preferences.recognitionLanguages.append(lang.code)
                            } else {
                                preferences.recognitionLanguages.removeAll { $0 == lang.code }
                            }
                        }
                    ))
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
