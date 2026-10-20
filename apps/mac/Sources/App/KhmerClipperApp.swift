import SwiftUI

@main
struct KhmerClipApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        // AppKit's NSApp.sendAction(Selector("showSettingsWindow:")) is blocked by the
        // system now ("Please use SettingsLink for opening the Settings scene"), and
        // SettingsLink only works from inside a SwiftUI view. This tiny always-alive
        // window gives the status-bar menu (AppKit) a SwiftUI context to call
        // openSettings() from, via SettingsOpenerRequest.post().
        Window("Hidden", id: "settings-opener") {
            SettingsOpenerView()
        }

        Settings {
            PreferencesView()
                .environmentObject(appDelegate.preferences)
        }
    }
}

private struct SettingsOpenerView: View {
    @Environment(\.openSettings) private var openSettings

    var body: some View {
        Color.clear
            .frame(width: 1, height: 1)
            .onAppear {
                // .windowStyle(.hiddenTitleBar) crashes AppKit's constraint system on
                // this window, so hide it via AppKit directly instead.
                NSApp.windows.first(where: { $0.title == "Hidden" })?.setIsVisible(false)
            }
            .onReceive(NotificationCenter.default.publisher(for: .settingsOpenerRequest)) { _ in
                NSApp.activate(ignoringOtherApps: true)
                openSettings()
            }
    }
}

extension Notification.Name {
    static let settingsOpenerRequest = Notification.Name("com.khmerclip.settingsOpenerRequest")
}
