import SwiftUI
import MASShortcut

/// Click-to-record control for assigning a global shortcut. Persistence and
/// live re-registration are handled by MASShortcutBinder via `defaultsKey`.
struct HotKeyRecorderView: NSViewRepresentable {
    let defaultsKey: String

    func makeNSView(context: Context) -> MASShortcutView {
        let view = MASShortcutView()
        view.style = .default
        view.associatedUserDefaultsKey = defaultsKey
        return view
    }

    func updateNSView(_ nsView: MASShortcutView, context: Context) {}
}
