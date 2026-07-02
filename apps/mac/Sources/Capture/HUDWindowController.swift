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
        let hosting = NSHostingController(rootView: StatusIconHUDView(isError: isError))
        let window = NSPanel(contentRect: NSRect(x: 0, y: 0, width: 64, height: 64),
                              styleMask: [.nonactivatingPanel, .titled, .fullSizeContentView],
                              backing: .buffered, defer: false)
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        window.isMovableByWindowBackground = false
        window.level = .floating
        window.backgroundColor = .clear
        window.isOpaque = false
        window.hasShadow = false
        window.ignoresMouseEvents = true
        window.contentViewController = hosting
        if let screen = NSScreen.main {
            let frame = window.frame
            window.setFrameOrigin(NSPoint(x: screen.frame.midX - frame.width / 2, y: screen.frame.minY + 80))
        }
        super.init(window: window)
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    private func showAndAutoDismiss() {
        window?.orderFrontRegardless()
        DispatchQueue.main.asyncAfter(deadline: .now() + 1) { [weak self] in
            self?.close()
        }
    }
}

private struct StatusIconHUDView: View {
    let isError: Bool

    var body: some View {
        Image(systemName: isError ? "xmark.circle.fill" : "checkmark.circle.fill")
            .resizable()
            .frame(width: 44, height: 44)
            .foregroundStyle(isError ? .red : .green)
            .padding(10)
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
