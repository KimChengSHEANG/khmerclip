import SwiftUI

@main
struct NetraApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        Settings {
            PreferencesView()
                .environmentObject(appDelegate.preferences)
        }
    }
}
