import Foundation
import Combine

final class AppPreferences: ObservableObject {
    @Published var autoCopyToClipboard: Bool {
        didSet { UserDefaults.standard.set(autoCopyToClipboard, forKey: Keys.autoCopy) }
    }
    @Published var showResultHUD: Bool {
        didSet { UserDefaults.standard.set(showResultHUD, forKey: Keys.showHUD) }
    }
    @Published var playCaptureSound: Bool {
        didSet { UserDefaults.standard.set(playCaptureSound, forKey: Keys.playSound) }
    }
    @Published var launchAtLogin: Bool {
        didSet { UserDefaults.standard.set(launchAtLogin, forKey: Keys.launchAtLogin) }
    }

    /// Defaults keys MASShortcutView/MASShortcutBinder read and write shortcuts under.
    enum ShortcutKeys {
        static let capture = "captureShortcut"
        static let captureFixture = "captureFixtureShortcut"
        static let qr = "qrShortcut"
    }

    private enum Keys {
        static let autoCopy = "autoCopyToClipboard"
        static let showHUD = "showResultHUD"
        static let playSound = "playCaptureSound"
        static let launchAtLogin = "launchAtLogin"
    }

    init() {
        let defaults = UserDefaults.standard
        self.autoCopyToClipboard = defaults.object(forKey: Keys.autoCopy) as? Bool ?? true
        self.showResultHUD = defaults.object(forKey: Keys.showHUD) as? Bool ?? true
        self.playCaptureSound = defaults.object(forKey: Keys.playSound) as? Bool ?? true
        self.launchAtLogin = defaults.object(forKey: Keys.launchAtLogin) as? Bool ?? false
    }
}
