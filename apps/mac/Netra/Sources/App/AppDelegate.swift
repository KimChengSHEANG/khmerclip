import AppKit
import SwiftUI

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    let preferences = AppPreferences()
    let historyStore = ScanHistoryStore()

    private var statusItem: NSStatusItem!
    private var coordinator: CaptureCoordinator!
    private var historyWindowController: NSWindowController?
    private var captureHotKeyID: UInt32?
    private var qrHotKeyID: UInt32?

    func applicationDidFinishLaunching(_ notification: Notification) {
        coordinator = CaptureCoordinator(preferences: preferences, historyStore: historyStore)
        setUpStatusItem()
        registerHotKeys()

        if preferences.launchAtLogin {
            LaunchAtLoginManager.setEnabled(true)
        }
    }

    private func setUpStatusItem() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        statusItem.button?.image = NSImage(systemSymbolName: "text.viewfinder", accessibilityDescription: "Netra")

        let menu = NSMenu()
        menu.addItem(withTitle: "Capture Text", action: #selector(captureText), keyEquivalent: "")
        menu.addItem(withTitle: "Scan QR / Barcode", action: #selector(captureBarcode), keyEquivalent: "")
        menu.addItem(withTitle: "OCR Image File…", action: #selector(showImageOCR), keyEquivalent: "")
        menu.addItem(.separator())
        menu.addItem(withTitle: "History…", action: #selector(showHistory), keyEquivalent: "")
        menu.addItem(withTitle: "Preferences…", action: #selector(showPreferences), keyEquivalent: ",")
        menu.addItem(.separator())
        menu.addItem(withTitle: "Quit Netra", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")

        for item in menu.items {
            item.target = self
        }
        statusItem.menu = menu
    }

    private func registerHotKeys() {
        if let id = captureHotKeyID { HotKeyManager.shared.unregister(id) }
        if let id = qrHotKeyID { HotKeyManager.shared.unregister(id) }

        captureHotKeyID = HotKeyManager.shared.register(preferences.captureHotKey) { [weak self] in
            self?.coordinator.beginCapture(mode: .text)
        }
        qrHotKeyID = HotKeyManager.shared.register(preferences.qrHotKey) { [weak self] in
            self?.coordinator.beginCapture(mode: .barcode)
        }
    }

    @objc private func captureText() {
        coordinator.beginCapture(mode: .text)
    }

    @objc private func captureBarcode() {
        coordinator.beginCapture(mode: .barcode)
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
        NSApp.activate(ignoringOtherApps: true)
        if #available(macOS 14.0, *) {
            NSApp.sendAction(Selector(("showSettingsWindow:")), to: nil, from: nil)
        }
    }
}
