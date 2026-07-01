import AppKit

/// Full-screen transparent window that lets the user drag out a capture rectangle,
/// mirroring the screenshot-selection UX of macOS's built-in tool and TextSniper.
final class SelectionOverlayWindow: NSWindow {
    init(screen: NSScreen) {
        super.init(contentRect: screen.frame, styleMask: [.borderless], backing: .buffered, defer: false)
        self.isOpaque = false
        self.backgroundColor = .clear
        self.level = .screenSaver
        self.ignoresMouseEvents = false
        self.hasShadow = false
        self.collectionBehavior = [.canJoinAllSpaces, .stationary, .fullScreenAuxiliary]
        self.setFrame(screen.frame, display: true)
    }

    override var canBecomeKey: Bool { true }
}

final class SelectionOverlayView: NSView {
    var onComplete: ((CGRect?) -> Void)?

    private var dragStart: NSPoint?
    private var currentRect: CGRect = .zero
    private let dimColor = NSColor.black.withAlphaComponent(0.25)

    override func draw(_ dirtyRect: NSRect) {
        dimColor.setFill()
        bounds.fill()

        guard currentRect != .zero else { return }
        NSColor.clear.setFill()
        currentRect.fill(using: .copy)

        NSColor.white.setStroke()
        let border = NSBezierPath(rect: currentRect)
        border.lineWidth = 1.5
        border.stroke()
    }

    override func mouseDown(with event: NSEvent) {
        dragStart = convert(event.locationInWindow, from: nil)
        currentRect = .zero
    }

    override func mouseDragged(with event: NSEvent) {
        guard let start = dragStart else { return }
        let point = convert(event.locationInWindow, from: nil)
        currentRect = NSRect(
            x: min(start.x, point.x),
            y: min(start.y, point.y),
            width: abs(point.x - start.x),
            height: abs(point.y - start.y)
        )
        needsDisplay = true
    }

    override func mouseUp(with event: NSEvent) {
        defer {
            dragStart = nil
            currentRect = .zero
        }
        guard currentRect.width > 4, currentRect.height > 4 else {
            onComplete?(nil)
            return
        }
        onComplete?(currentRect)
    }

    override func keyDown(with event: NSEvent) {
        if event.keyCode == 53 { // Escape
            onComplete?(nil)
        }
    }

    override var acceptsFirstResponder: Bool { true }
}
