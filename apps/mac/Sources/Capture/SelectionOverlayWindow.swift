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
    private let dimColor = NSColor.clear
    private var mouseLocation: NSPoint = .zero

    /// Space bar toggles "click a window" mode, same as macOS's own screenshot tool.
    private var isWindowMode = false
    private var pickableWindows: [PickableWindow] = []
    private var hoveredWindowRect: CGRect?

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        addTrackingArea(NSTrackingArea(rect: frameRect, options: [.mouseMoved, .activeAlways, .inVisibleRect], owner: self))
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    /// Primes the reticle/coordinate readout with the cursor's current position so they
    /// render immediately on activation instead of waiting for the first mouse move.
    func primeMouseLocation(_ locationInWindow: NSPoint) {
        mouseLocation = convert(locationInWindow, from: nil)
        needsDisplay = true
    }

    override func draw(_ dirtyRect: NSRect) {
        dimColor.setFill()
        bounds.fill()

        if isWindowMode {
            if let rect = hoveredWindowRect {
                NSColor.clear.setFill()
                rect.fill(using: .copy)
                NSColor.systemBlue.setStroke()
                let border = NSBezierPath(rect: rect)
                border.lineWidth = 2
                border.stroke()
            }
        } else if currentRect != .zero {
            NSColor.black.withAlphaComponent(0.3).setFill()
            currentRect.fill(using: .copy)

            NSColor.white.setStroke()
            let border = NSBezierPath(rect: currentRect)
            border.lineWidth = 1.5
            border.stroke()

            drawLabel(String(format: "%.0f × %.0f", currentRect.width, currentRect.height), near: mouseLocation)
        }

        drawCrosshairCursor()
        if !isWindowMode && currentRect == .zero {
            drawCoordinateNumbers()
        }
    }

    override func mouseMoved(with event: NSEvent) {
        mouseLocation = convert(event.locationInWindow, from: nil)
        if isWindowMode {
            updateHoveredWindow(at: event.locationInWindow)
        } else {
            needsDisplay = true
        }
    }

    /// Draws a target-reticle cursor (circle + crosshair) since the real cursor is hidden during capture.
    private func drawCrosshairCursor() {
        let armLength: CGFloat = 12
        let radius: CGFloat = 5

        NSGraphicsContext.saveGraphicsState()
        let shadow = NSShadow()
        shadow.shadowColor = NSColor.white.withAlphaComponent(0.9)
        shadow.shadowBlurRadius = 1.5
        shadow.shadowOffset = .zero
        shadow.set()

        NSColor.black.setStroke()
        let plus = NSBezierPath()
        plus.lineWidth = 1.2
        plus.move(to: NSPoint(x: mouseLocation.x - armLength, y: mouseLocation.y))
        plus.line(to: NSPoint(x: mouseLocation.x + armLength, y: mouseLocation.y))
        plus.move(to: NSPoint(x: mouseLocation.x, y: mouseLocation.y - armLength))
        plus.line(to: NSPoint(x: mouseLocation.x, y: mouseLocation.y + armLength))
        plus.stroke()

        NSGraphicsContext.restoreGraphicsState()

        // let circleRect = NSRect(x: mouseLocation.x - radius, y: mouseLocation.y - radius, width: radius * 2, height: radius * 2)
        // let circle = NSBezierPath(ovalIn: circleRect)
        // NSColor.white.setFill()
        // circle.fill()
        // circle.lineWidth = 1.5
        // circle.stroke()
    }

    /// Shows the pointer's screen-pixel coordinates stacked below the reticle, mirroring macOS's own screenshot tool.
    private func drawCoordinateNumbers() {
        guard let screen = window?.screen else { return }
        let globalAppKitPoint = NSPoint(x: screen.frame.origin.x + mouseLocation.x, y: screen.frame.origin.y + mouseLocation.y)
        let primaryHeight = NSScreen.screens.first(where: { $0.frame.origin == .zero })?.frame.height ?? screen.frame.height
        let x = Int(globalAppKitPoint.x)
        let y = Int(primaryHeight - globalAppKitPoint.y)

        let textShadow = NSShadow()
        textShadow.shadowColor = NSColor.white.withAlphaComponent(0.9)
        textShadow.shadowBlurRadius = 1.5
        textShadow.shadowOffset = .zero

        let attributed = NSAttributedString(string: "\(x)\n\(y)", attributes: [
            .font: NSFont.boldSystemFont(ofSize: 13),
            .foregroundColor: NSColor.black,
            .shadow: textShadow
        ])
        let size = attributed.size()
        attributed.draw(at: NSPoint(x: mouseLocation.x + 10, y: mouseLocation.y - size.height - 2))
    }

    private func drawLabel(_ text: String, near point: NSPoint) {
        let attributed = NSAttributedString(string: text, attributes: [
            .font: NSFont.monospacedDigitSystemFont(ofSize: 12, weight: .medium),
            .foregroundColor: NSColor.white
        ])
        let padding = NSSize(width: 8, height: 4)
        let textSize = attributed.size()
        let origin = NSPoint(x: point.x + 14, y: point.y + 14)
        let boxRect = NSRect(x: origin.x, y: origin.y, width: textSize.width + padding.width * 2, height: textSize.height + padding.height * 2)

        NSColor.black.withAlphaComponent(0.75).setFill()
        NSBezierPath(roundedRect: boxRect, xRadius: 4, yRadius: 4).fill()
        attributed.draw(at: NSPoint(x: boxRect.minX + padding.width, y: boxRect.minY + padding.height))
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
        mouseLocation = point
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

    /// Deliver the very first click to the view even when the overlay isn't the active
    /// window yet — otherwise macOS eats the first mouseDown just to activate the window,
    /// forcing the user to click-drag a second time.
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }
}

private extension NSPoint {
    func subtracting(_ other: NSPoint) -> NSPoint {
        NSPoint(x: x - other.x, y: y - other.y)
    }
}
