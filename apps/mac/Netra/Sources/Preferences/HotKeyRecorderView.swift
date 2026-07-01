import SwiftUI
import Carbon.HIToolbox

/// Click-to-record control for assigning a global shortcut, the same
/// interaction pattern System Settings and Alfred use for hotkey fields.
struct HotKeyRecorderView: NSViewRepresentable {
    @Binding var definition: HotKeyDefinition

    func makeNSView(context: Context) -> RecorderButton {
        let button = RecorderButton()
        button.onRecorded = { keyCode, modifiers in
            definition = HotKeyDefinition(keyCode: keyCode, modifiers: modifiers)
        }
        return button
    }

    func updateNSView(_ nsView: RecorderButton, context: Context) {
        nsView.title = definition.displayString
    }
}

final class RecorderButton: NSButton {
    var onRecorded: ((UInt32, UInt32) -> Void)?
    private var isRecording = false
    private var monitor: Any?

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        bezelStyle = .rounded
        setButtonType(.momentaryPushIn)
        target = self
        action = #selector(startRecording)
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    @objc private func startRecording() {
        guard !isRecording else { return }
        isRecording = true
        title = "Press keys…"
        monitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            guard let self else { return event }
            self.finishRecording(keyCode: UInt32(event.keyCode), modifierFlags: event.modifierFlags)
            return nil
        }
    }

    private func finishRecording(keyCode: UInt32, modifierFlags: NSEvent.ModifierFlags) {
        var carbonModifiers: UInt32 = 0
        if modifierFlags.contains(.command) { carbonModifiers |= UInt32(cmdKey) }
        if modifierFlags.contains(.option) { carbonModifiers |= UInt32(optionKey) }
        if modifierFlags.contains(.shift) { carbonModifiers |= UInt32(shiftKey) }
        if modifierFlags.contains(.control) { carbonModifiers |= UInt32(controlKey) }

        isRecording = false
        if let monitor {
            NSEvent.removeMonitor(monitor)
        }
        monitor = nil
        onRecorded?(keyCode, carbonModifiers)
    }

    deinit {
        if let monitor {
            NSEvent.removeMonitor(monitor)
        }
    }
}
