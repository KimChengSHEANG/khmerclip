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
    @Published var ocrEngine: OCREngineKind {
        didSet { UserDefaults.standard.set(ocrEngine.rawValue, forKey: Keys.ocrEngine) }
    }
    @Published var preserveLinebreaks: Bool {
        didSet { UserDefaults.standard.set(preserveLinebreaks, forKey: Keys.preserveLinebreaks) }
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
        static let ocrEngine = "ocrEngine"
        static let preserveLinebreaks = "preserveLinebreaks"
    }

    init() {
        let defaults = UserDefaults.standard
        self.autoCopyToClipboard = defaults.object(forKey: Keys.autoCopy) as? Bool ?? true
        self.showResultHUD = defaults.object(forKey: Keys.showHUD) as? Bool ?? true
        self.playCaptureSound = defaults.object(forKey: Keys.playSound) as? Bool ?? true
        self.launchAtLogin = defaults.object(forKey: Keys.launchAtLogin) as? Bool ?? false
        self.ocrEngine = (defaults.string(forKey: Keys.ocrEngine)).flatMap(OCREngineKind.init) ?? .tesseract
        self.preserveLinebreaks = defaults.object(forKey: Keys.preserveLinebreaks) as? Bool ?? true
    }
}
