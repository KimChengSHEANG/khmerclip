import AppKit
import SwiftUI
import Sparkle
import MASShortcut
import Carbon.HIToolbox

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate, NSMenuDelegate {
    let preferences = AppPreferences()
    let historyStore = ScanHistoryStore()

    private var statusItem: NSStatusItem!
    private var coordinator: CaptureCoordinator!
    private var historyWindowController: NSWindowController?
    private var captureMenuItem: NSMenuItem!
    private var qrMenuItem: NSMenuItem!
    /// Auto-update via Sparkle. SUFeedURL in Info.plist is a placeholder until
    /// this app has a real hosted appcast — see README "Auto-update" section.
    private let updaterController = SPUStandardUpdaterController(startingUpdater: true, updaterDelegate: nil, userDriverDelegate: nil)

    func applicationDidFinishLaunching(_ notification: Notification) {
        coordinator = CaptureCoordinator(preferences: preferences, historyStore: historyStore)
        setUpStatusItem()
        registerHotKeys()

        if preferences.launchAtLogin {
            LaunchAtLoginManager.setEnabled(true)
        }

        if OnboardingWindowPresenter.shouldShow {
            OnboardingWindowPresenter.show()
        }
    }

    private func setUpStatusItem() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        statusItem.button?.image = NSImage(systemSymbolName: "text.viewfinder", accessibilityDescription: "Netra")

        let menu = NSMenu()
        menu.delegate = self
        captureMenuItem = menu.addItem(withTitle: "Capture Text", action: #selector(captureText), keyEquivalent: "")
        qrMenuItem = menu.addItem(withTitle: "Scan QR / Barcode", action: #selector(captureBarcode), keyEquivalent: "")
        menu.addItem(withTitle: "OCR Image File…", action: #selector(showImageOCR), keyEquivalent: "")
        menu.addItem(.separator())
        menu.addItem(withTitle: "History…", action: #selector(showHistory), keyEquivalent: "")
        menu.addItem(withTitle: "Preferences…", action: #selector(showPreferences), keyEquivalent: ",")
        menu.addItem(withTitle: "Welcome / Permissions…", action: #selector(showOnboarding), keyEquivalent: "")
        menu.addItem(.separator())
        let updateItem = menu.addItem(withTitle: "Check for Updates…", action: #selector(SPUStandardUpdaterController.checkForUpdates(_:)), keyEquivalent: "")
        updateItem.target = updaterController
        menu.addItem(.separator())
        let quitItem = menu.addItem(withTitle: "Quit Netra", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        quitItem.target = NSApp

        for item in menu.items where item.target == nil {
            item.target = self
        }
        statusItem.menu = menu
    }

    private func registerHotKeys() {
        MASShortcutBinder.shared().registerDefaultShortcuts([
            AppPreferences.ShortcutKeys.capture: MASShortcut(keyCode: Int(kVK_ANSI_5), modifierFlags: [.command, .shift]),
            AppPreferences.ShortcutKeys.qr: MASShortcut(keyCode: Int(kVK_ANSI_6), modifierFlags: [.command, .shift]),
        ])
        MASShortcutBinder.shared().bindShortcut(withDefaultsKey: AppPreferences.ShortcutKeys.capture) { [weak self] in
            self?.coordinator.beginCapture(mode: .text)
        }
        MASShortcutBinder.shared().bindShortcut(withDefaultsKey: AppPreferences.ShortcutKeys.qr) { [weak self] in
            self?.coordinator.beginCapture(mode: .barcode)
        }
    }

    /// Keeps the menu's key equivalents in sync with whatever the user last set in
    /// Preferences; refreshed on every open rather than once, since MASShortcutBinder
    /// doesn't offer a change callback we can hook here.
    func menuNeedsUpdate(_ menu: NSMenu) {
        applyShortcut(forKey: AppPreferences.ShortcutKeys.capture, to: captureMenuItem)
        applyShortcut(forKey: AppPreferences.ShortcutKeys.qr, to: qrMenuItem)
    }

    private func applyShortcut(forKey key: String, to item: NSMenuItem) {
        guard let data = UserDefaults.standard.data(forKey: key),
              let shortcut = try? NSKeyedUnarchiver.unarchivedObject(ofClass: MASShortcut.self, from: data),
              let keyEquivalent = shortcut.keyCodeStringForKeyEquivalent else {
            item.keyEquivalent = ""
            item.keyEquivalentModifierMask = []
            return
        }
        item.keyEquivalent = keyEquivalent.lowercased()
        item.keyEquivalentModifierMask = shortcut.modifierFlags
    }

    @objc private func captureText() {
        coordinator.beginCapture(mode: .text)
    }

    @objc private func captureBarcode() {
        coordinator.beginCapture(mode: .barcode)
    }

    @objc private func showOnboarding() {
        OnboardingWindowPresenter.show()
    }

    @objc private func showImageOCR() {
        ImageOCRWindowPresenter.show(coordinator: coordinator)
    }

    @objc private func showHistory() {
        if historyWindowController == nil {
            let hosting = NSHostingController(rootView: HistoryView(store: historyStore))
            let window = NSWindow(contentViewController: hosting)
            window.title = "History"
            window.styleMask = [.titled, .closable, .miniaturizable]
            window.center()
            historyWindowController = NSWindowController(window: window)
        }
        historyWindowController?.showWindow(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    @objc private func showPreferences() {
        NotificationCenter.default.post(name: .settingsOpenerRequest, object: nil)
    }

    // Netra has no Dock icon and lives entirely in the status bar; closing/hiding
    // its windows (including the hidden Settings-opener window) must never quit it.
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false
    }
}
