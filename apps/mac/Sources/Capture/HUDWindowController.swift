import SwiftUI
import AppKit

/// Small floating panel shown near the top of the screen right after a capture,
/// mirroring TextSniper's transient result HUD.
@MainActor
final class HUDWindowController: NSWindowController {
    private static var current: HUDWindowController?

    @discardableResult
    static func showResult(text: String) -> HUDWindowController {
        current?.close()
        let controller = HUDWindowController(text: text, isError: false)
        current = controller
        controller.showAndAutoDismiss()
        return controller
    }

    @discardableResult
    static func showMessage(_ message: String) -> HUDWindowController {
        current?.close()
        let controller = HUDWindowController(text: message, isError: true)
        current = controller
        controller.showAndAutoDismiss()
        return controller
    }

    private init(text: String, isError: Bool) {
        let hosting = NSHostingController(rootView: ResultHUDView(text: text, isError: isError))
        let window = NSPanel(contentRect: NSRect(x: 0, y: 0, width: 360, height: 140),
                              styleMask: [.nonactivatingPanel, .titled, .fullSizeContentView],
                              backing: .buffered, defer: false)
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        window.isMovableByWindowBackground = true
        window.level = .floating
        window.contentViewController = hosting
        window.center()
        if let screen = NSScreen.main {
            let frame = window.frame
            window.setFrameOrigin(NSPoint(x: screen.frame.midX - frame.width / 2, y: screen.frame.maxY - frame.height - 80))
        }
        super.init(window: window)
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    private func showAndAutoDismiss() {
        window?.makeKeyAndOrderFront(nil)
        DispatchQueue.main.asyncAfter(deadline: .now() + 6) { [weak self] in
            self?.close()
        }
    }
}

private struct ResultHUDView: View {
    let text: String
    let isError: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: isError ? "exclamationmark.triangle" : "checkmark.circle.fill")
                    .foregroundStyle(isError ? .orange : .green)
                Text(isError ? "Netra" : "Text Copied")
                    .font(.headline)
                Spacer()
            }
            ScrollView {
                Text(text)
                    .font(.system(size: 12))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 70)

            if !isError {
                HStack {
                    Button("Speak") { TextToSpeechManager.shared.speak(text) }
                    Button("Translate") { openTranslateWindow(text: text) }
                    Spacer()
                }
                .font(.caption)
            }
        }
        .padding(12)
        .frame(width: 360)
    }

    private func openTranslateWindow(text: String) {
        TranslateWindowPresenter.show(text: text)
    }
}

/// Keeps a strong reference to the translate window so it isn't deallocated
/// the moment the presenting closure returns.
@MainActor
enum TranslateWindowPresenter {
    private static var controller: NSWindowController?

    static func show(text: String) {
        let hosting = NSHostingController(rootView: TranslateView(sourceText: text))
        let window = NSWindow(contentViewController: hosting)
        window.title = "Translate"
        window.styleMask = [.titled, .closable]
        window.center()
        let windowController = NSWindowController(window: window)
        controller = windowController
        windowController.showWindow(nil)
        NSApp.activate(ignoringOtherApps: true)
    }
}
