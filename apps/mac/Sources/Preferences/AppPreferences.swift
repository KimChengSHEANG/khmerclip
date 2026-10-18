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
    /// Keep each capture's image with its history entry so wrong results can be flagged.
    @Published var keepCaptureImages: Bool {
        didSet { UserDefaults.standard.set(keepCaptureImages, forKey: Keys.keepCaptureImages) }
    }

    /// Send crash reports to the developer automatically (anonymous; no images or scanned text).
    @Published var sendCrashReports: Bool {
        didSet { UserDefaults.standard.set(sendCrashReports, forKey: Keys.sendCrashReports) }
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
        static let keepCaptureImages = "keepCaptureImages"
        static let sendCrashReports = "sendCrashReports"
    }

    init() {
        let defaults = UserDefaults.standard
        self.autoCopyToClipboard = defaults.object(forKey: Keys.autoCopy) as? Bool ?? true
        self.showResultHUD = defaults.object(forKey: Keys.showHUD) as? Bool ?? true
        self.playCaptureSound = defaults.object(forKey: Keys.playSound) as? Bool ?? true
        self.launchAtLogin = defaults.object(forKey: Keys.launchAtLogin) as? Bool ?? false
        self.ocrEngine = (defaults.string(forKey: Keys.ocrEngine)).flatMap(OCREngineKind.init) ?? .tesseract
        self.preserveLinebreaks = defaults.object(forKey: Keys.preserveLinebreaks) as? Bool ?? true
        self.keepCaptureImages = defaults.object(forKey: Keys.keepCaptureImages) as? Bool ?? true
        self.sendCrashReports = defaults.object(forKey: Keys.sendCrashReports) as? Bool ?? true
    }
}
