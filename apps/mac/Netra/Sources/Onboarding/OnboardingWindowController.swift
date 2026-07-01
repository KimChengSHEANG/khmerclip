import SwiftUI
import AppKit
import CoreGraphics

/// First-launch explainer + Screen Recording permission request. Shown once
/// (tracked via UserDefaults) before the status item becomes useful, since a
/// menu-bar-only (LSUIElement) app has no other natural place to greet the user.
enum OnboardingWindowPresenter {
    private static var controller: NSWindowController?
    private static let completedKey = "hasCompletedOnboarding"

    static var shouldShow: Bool {
        !UserDefaults.standard.bool(forKey: completedKey)
    }

    static func show() {
        let hosting = NSHostingController(rootView: OnboardingView(onFinish: {
            UserDefaults.standard.set(true, forKey: completedKey)
            controller?.close()
            controller = nil
        }))
        let window = NSWindow(contentViewController: hosting)
        window.title = "Welcome to Netra"
        window.styleMask = [.titled, .closable]
        window.center()
        window.level = .floating
        let windowController = NSWindowController(window: window)
        controller = windowController
        windowController.showWindow(nil)
        NSApp.activate(ignoringOtherApps: true)
    }
}

struct OnboardingView: View {
    let onFinish: () -> Void

    @State private var hasScreenRecordingAccess = CGPreflightScreenCaptureAccess()

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(spacing: 12) {
                Image(systemName: "text.viewfinder")
                    .font(.system(size: 36))
                    .foregroundStyle(.blue)
                VStack(alignment: .leading) {
                    Text("Welcome to Netra").font(.title2).bold()
                    Text("Khmer OCR, live from your screen.").foregroundStyle(.secondary)
                }
            }

            Divider()

            VStack(alignment: .leading, spacing: 8) {
                Label("Lives in your menu bar — look for the scan icon.", systemImage: "menubar.rectangle")
                Label("Press your capture hotkey, drag a box, get text on your clipboard.", systemImage: "selection.pin.in.out")
                Label("Khmer OCR runs on a bundled engine; other languages use Apple's Vision.", systemImage: "character.book.closed")
            }
            .font(.callout)

            Divider()

            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Image(systemName: hasScreenRecordingAccess ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                        .foregroundStyle(hasScreenRecordingAccess ? .green : .orange)
                    Text("Screen Recording access").bold()
                }
                Text("Netra needs this to capture the region you select. macOS will prompt on first use — or grant it now:")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                HStack {
                    Button("Open System Settings…") { openScreenRecordingSettings() }
                    Button("Re-check") { hasScreenRecordingAccess = CGPreflightScreenCaptureAccess() }
                }
            }

            Spacer()

            HStack {
                Spacer()
                Button("Get Started") { onFinish() }
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(24)
        .frame(width: 420, height: 380)
    }

    private func openScreenRecordingSettings() {
        guard let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture") else { return }
        NSWorkspace.shared.open(url)
    }
}
