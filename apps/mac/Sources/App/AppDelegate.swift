import AppKit
import Combine
import SwiftUI
import MASShortcut
import Carbon.HIToolbox

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate, NSMenuDelegate {
    let preferences = AppPreferences()
    let historyStore = ScanHistoryStore()

    private var statusItem: NSStatusItem!
    private var coordinator: CaptureCoordinator!
    private var historyWindowController: NSWindowController?
    private var feedbackWindowController: NSWindowController?
    private var captureMenuItem: NSMenuItem!
    private var preserveLinebreaksMenuItem: NSMenuItem!
    private var keepCapturesSubscription: AnyCancellable?
    /// "Check for Updates…" opens the GitHub release page; the app never updates itself.
    static let releasesURL = URL(string: "https://github.com/KimChengSHEANG/khmerclip/releases/latest")!

    func applicationDidFinishLaunching(_ notification: Notification) {
        let crashPending = ScanHistoryStore.dataDir.appendingPathComponent("crash-pending.txt")
        CrashReporter.collect(pendingURL: crashPending, outbox: ReportOutbox.shared,
                              engine: preferences.ocrEngine.rawValue, enabled: preferences.sendCrashReports)
        CrashReporter.install(pendingURL: crashPending)
        Task { _ = await ReportOutbox.shared.flush(post: { await ReportOutbox.post($0, to: ReportOutbox.reportURL) }) }
        coordinator = CaptureCoordinator(preferences: preferences, historyStore: historyStore)
        setUpStatusItem()
        registerHotKeys()
        // Turning "Keep capture images" off deletes the images already kept.
        keepCapturesSubscription = preferences.$keepCaptureImages.dropFirst().filter { !$0 }
            .sink { [historyStore] _ in historyStore.purgeCaptures() }

        coordinator.onOCRProgress = { [weak self] info in
            OCRProgressController.update(info: info) { self?.coordinator.cancelCapture() }
        }

        if preferences.launchAtLogin {
            LaunchAtLoginManager.setEnabled(true)
        }

        if OnboardingWindowPresenter.shouldShow {
            OnboardingWindowPresenter.show()
        }
    }

    private func setUpStatusItem() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        statusItem.button?.image = NSImage(systemSymbolName: "text.viewfinder", accessibilityDescription: "KhmerClip")

        let menu = NSMenu()
        menu.delegate = self
        captureMenuItem = menu.addItem(withTitle: "Capture Text", action: #selector(captureText), keyEquivalent: "")
        menu.addItem(.separator())
        preserveLinebreaksMenuItem = menu.addItem(withTitle: "Preserve Linebreaks", action: #selector(togglePreserveLinebreaks), keyEquivalent: "")
        menu.addItem(.separator())
        menu.addItem(withTitle: "History…", action: #selector(showHistory), keyEquivalent: "")
        menu.addItem(withTitle: "Send Feedback…", action: #selector(showFeedback), keyEquivalent: "")
        menu.addItem(withTitle: "Preferences…", action: #selector(showPreferences), keyEquivalent: ",")
        menu.addItem(withTitle: "Welcome / Permissions…", action: #selector(showOnboarding), keyEquivalent: "")
        menu.addItem(.separator())
        menu.addItem(withTitle: "Check for Updates…", action: #selector(openReleases), keyEquivalent: "")
        menu.addItem(.separator())
        let quitItem = menu.addItem(withTitle: "Quit KhmerClip", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        quitItem.target = NSApp

        for item in menu.items where item.target == nil {
            item.target = self
        }
        statusItem.menu = menu
    }

    private func registerHotKeys() {
        MASShortcutBinder.shared().registerDefaultShortcuts([
            AppPreferences.ShortcutKeys.capture: MASShortcut(keyCode: Int(kVK_ANSI_6), modifierFlags: [.command, .shift]),
        ])
        MASShortcutBinder.shared().bindShortcut(withDefaultsKey: AppPreferences.ShortcutKeys.capture) { [weak self] in
            self?.coordinator.beginCapture(mode: .text)
        }
        #if DEBUG
        // Dev-only: capture + dump the result into this checkout's Tests/Fixtures/ocr as a new
        // case. Release builds skip it: the path is the build machine's, and it would claim ⌘⇧7.
        MASShortcutBinder.shared().registerDefaultShortcuts([
            AppPreferences.ShortcutKeys.captureFixture: MASShortcut(keyCode: Int(kVK_ANSI_7), modifierFlags: [.command, .shift]),
        ])
        MASShortcutBinder.shared().bindShortcut(withDefaultsKey: AppPreferences.ShortcutKeys.captureFixture) { [weak self] in
            self?.coordinator.beginCapture(mode: .text, saveFixture: true)
        }
        #endif
    }

    /// Keeps the menu's key equivalents in sync with whatever the user last set in
    /// Preferences; refreshed on every open rather than once, since MASShortcutBinder
    /// doesn't offer a change callback we can hook here.
    func menuNeedsUpdate(_ menu: NSMenu) {
        applyShortcut(forKey: AppPreferences.ShortcutKeys.capture, to: captureMenuItem)
        preserveLinebreaksMenuItem.state = preferences.preserveLinebreaks ? .on : .off
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

    @objc private func togglePreserveLinebreaks() {
        preferences.preserveLinebreaks.toggle()
    }

    @objc private func showOnboarding() {
        OnboardingWindowPresenter.show()
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

    @objc private func showFeedback() {
        if feedbackWindowController == nil {
            let hosting = NSHostingController(rootView: FeedbackView(onSend: { [weak self] in self?.sendFeedback($0, email: $1) }))
            let window = NSWindow(contentViewController: hosting)
            window.title = "Send Feedback"
            window.styleMask = [.titled, .closable]
            window.isReleasedWhenClosed = false
            window.center()
            feedbackWindowController = NSWindowController(window: window)
        }
        feedbackWindowController?.showWindow(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    /// Closes the window only once the report is safely queued; on failure it stays open with the text intact.
    private func sendFeedback(_ message: String, email: String) {
        let payload = Report.feedback(message: message, email: email, engine: preferences.ocrEngine.rawValue)
        do {
            try ReportSender.queueAndSend(payload) { [weak self] in self?.showReportAlert($0) }
            feedbackWindowController?.close()
            feedbackWindowController = nil  // next open starts with a blank form
        } catch {
            showReportAlert(ReportSender.saveFailedMessage)
        }
    }

    private func showReportAlert(_ message: String) {
        let alert = NSAlert()
        alert.messageText = message
        alert.addButton(withTitle: "OK")
        NSApp.activate(ignoringOtherApps: true)
        alert.runModal()
    }

    @objc private func showPreferences() {
        NotificationCenter.default.post(name: .settingsOpenerRequest, object: nil)
    }

    // KhmerClip has no Dock icon and lives entirely in the status bar; closing/hiding
    // its windows (including the hidden Settings-opener window) must never quit it.
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false
    }

    @objc private func openReleases() {
        NSWorkspace.shared.open(Self.releasesURL)
    }
}
