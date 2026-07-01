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
        self.acceptsMouseMovedEvents = true
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

    /// Space bar toggles "click a window" mode, same as macOS's own screenshot tool.
    private var isWindowMode = false
    private var pickableWindows: [PickableWindow] = []
    private var hoveredWindowRect: CGRect?

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        addTrackingArea(NSTrackingArea(rect: frameRect, options: [.mouseMoved, .activeAlways, .inVisibleRect], owner: self))
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    override func draw(_ dirtyRect: NSRect) {
        dimColor.setFill()
        bounds.fill()

        if isWindowMode {
            guard let rect = hoveredWindowRect else { return }
            NSColor.clear.setFill()
            rect.fill(using: .copy)
            NSColor.systemBlue.setStroke()
            let border = NSBezierPath(rect: rect)
            border.lineWidth = 2
            border.stroke()
            return
        }

        guard currentRect != .zero else { return }
        NSColor.clear.setFill()
        currentRect.fill(using: .copy)

        NSColor.white.setStroke()
        let border = NSBezierPath(rect: currentRect)
        border.lineWidth = 1.5
        border.stroke()
    }

    override func mouseMoved(with event: NSEvent) {
        guard isWindowMode else { return }
        updateHoveredWindow(at: event.locationInWindow)
    }

    private func updateHoveredWindow(at locationInWindow: NSPoint) {
        guard let screen = window?.screen else { return }
        let localPoint = convert(locationInWindow, from: nil)
        let globalAppKitPoint = NSPoint(x: screen.frame.origin.x + localPoint.x, y: screen.frame.origin.y + localPoint.y)
        let primaryHeight = NSScreen.screens.first(where: { $0.frame.origin == .zero })?.frame.height ?? screen.frame.height
        let globalCGPoint = CGPoint(x: globalAppKitPoint.x, y: primaryHeight - globalAppKitPoint.y)

        if let hit = WindowPicker.window(at: globalCGPoint, in: pickableWindows) {
            let appKitGlobalRect = WindowPicker.appKitRect(from: hit.bounds, primaryScreenHeight: primaryHeight)
            hoveredWindowRect = CGRect(
                x: appKitGlobalRect.origin.x - screen.frame.origin.x,
                y: appKitGlobalRect.origin.y - screen.frame.origin.y,
                width: appKitGlobalRect.width,
                height: appKitGlobalRect.height
            )
        } else {
            hoveredWindowRect = nil
        }
        needsDisplay = true
    }

    override func mouseDown(with event: NSEvent) {
        if isWindowMode {
            updateHoveredWindow(at: event.locationInWindow)
            return
        }
        dragStart = convert(event.locationInWindow, from: nil)
        currentRect = .zero
    }

    override func mouseDragged(with event: NSEvent) {
        guard !isWindowMode, let start = dragStart else { return }
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
        if isWindowMode {
            onComplete?(hoveredWindowRect)
            return
        }
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
        switch event.keyCode {
        case 53: // Escape
            onComplete?(nil)
        case 49: // Space
            isWindowMode.toggle()
            if isWindowMode {
                pickableWindows = WindowPicker.onScreenWindows()
                updateHoveredWindow(at: NSEvent.mouseLocation.subtracting(window?.frame.origin ?? .zero))
            } else {
                hoveredWindowRect = nil
            }
            needsDisplay = true
        default:
            break
        }
    }

    override var acceptsFirstResponder: Bool { true }
}

private extension NSPoint {
    func subtracting(_ other: NSPoint) -> NSPoint {
        NSPoint(x: x - other.x, y: y - other.y)
    }
}
