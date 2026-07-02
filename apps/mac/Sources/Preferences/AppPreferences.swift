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
    @Published var captureHotKey: HotKeyDefinition {
        didSet { HotKeyDefinition.save(captureHotKey, key: Keys.captureHotKey) }
    }
    @Published var qrHotKey: HotKeyDefinition {
        didSet { HotKeyDefinition.save(qrHotKey, key: Keys.qrHotKey) }
    }

    private enum Keys {
        static let autoCopy = "autoCopyToClipboard"
        static let showHUD = "showResultHUD"
        static let playSound = "playCaptureSound"
        static let launchAtLogin = "launchAtLogin"
        static let captureHotKey = "captureHotKey"
        static let qrHotKey = "qrHotKey"
    }

    init() {
        let defaults = UserDefaults.standard
        self.autoCopyToClipboard = defaults.object(forKey: Keys.autoCopy) as? Bool ?? true
        self.showResultHUD = defaults.object(forKey: Keys.showHUD) as? Bool ?? true
        self.playCaptureSound = defaults.object(forKey: Keys.playSound) as? Bool ?? true
        self.launchAtLogin = defaults.object(forKey: Keys.launchAtLogin) as? Bool ?? false
        self.captureHotKey = HotKeyDefinition.load(key: Keys.captureHotKey) ?? .defaultCapture
        self.qrHotKey = HotKeyDefinition.load(key: Keys.qrHotKey) ?? .defaultQR
    }
}
