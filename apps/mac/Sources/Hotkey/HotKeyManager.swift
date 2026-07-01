import Carbon.HIToolbox
import AppKit

/// Registers global keyboard shortcuts via the Carbon Event Manager, the standard
/// mechanism system-wide utilities (Alfred, Rectangle, TextSniper) use for hotkeys
/// that fire even when the app is not frontmost.
@MainActor
final class HotKeyManager {
    typealias Handler = () -> Void

    private var hotKeyRefs: [UInt32: EventHotKeyRef] = [:]
    private var handlers: [UInt32: Handler] = [:]
    private var nextID: UInt32 = 1
    private var eventHandler: EventHandlerRef?

    static let shared = HotKeyManager()

    private init() {
        installEventHandler()
    }

    private func installEventHandler() {
        var eventType = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: OSType(kEventHotKeyPressed))
        InstallEventHandler(GetApplicationEventTarget(), { _, event, userData in
            guard let event, let userData else { return noErr }
            var hotKeyID = EventHotKeyID()
            GetEventParameter(event, EventParamName(kEventParamDirectObject), EventParamType(typeEventHotKeyID), nil, MemoryLayout<EventHotKeyID>.size, nil, &hotKeyID)
            let manager = Unmanaged<HotKeyManager>.fromOpaque(userData).takeUnretainedValue()
            manager.handlers[hotKeyID.id]?()
            return noErr
        }, 1, &eventType, Unmanaged.passUnretained(self).toOpaque(), &eventHandler)
    }

    @discardableResult
    func register(_ definition: HotKeyDefinition, handler: @escaping Handler) -> UInt32 {
        let id = nextID
        nextID += 1
        var hotKeyRef: EventHotKeyRef?
        let hotKeyID = EventHotKeyID(signature: OSType(0x4E545241), id: id) // 'NTRA'
        let status = RegisterEventHotKey(definition.keyCode, definition.modifiers, hotKeyID, GetApplicationEventTarget(), 0, &hotKeyRef)
        if status == noErr, let ref = hotKeyRef {
            hotKeyRefs[id] = ref
            handlers[id] = handler
        }
        return id
    }

    func unregister(_ id: UInt32) {
        if let ref = hotKeyRefs[id] {
            UnregisterEventHotKey(ref)
        }
        hotKeyRefs.removeValue(forKey: id)
        handlers.removeValue(forKey: id)
    }

    func unregisterAll() {
        for id in Array(hotKeyRefs.keys) {
            unregister(id)
        }
    }
}
