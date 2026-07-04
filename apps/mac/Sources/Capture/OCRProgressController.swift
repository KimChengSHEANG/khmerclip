import SwiftUI
import AppKit

/// Floating progress box shown near the bottom of the screen while OCR runs,
/// with a phase label, percentage, and a Cancel button. TrOCR's decode loop can
/// take several seconds, so this gives live feedback plus a way to bail out.
@MainActor
final class OCRProgressController: NSWindowController {
    private static var current: OCRProgressController?
    private let model = OCRProgressModel()

    /// Drives the panel from a single callback: a value shows/updates the box,
    /// `nil` tears it down. `onCancel` is wired on first show.
    static func update(info: OCRProgress?, onCancel: @escaping () -> Void) {
        guard let info else {
            current?.close()
            current = nil
            return
        }
        if let current {
            current.model.info = info
        } else {
            let controller = OCRProgressController(onCancel: onCancel)
            current = controller
            controller.model.info = info
            controller.window?.orderFrontRegardless()
        }
    }

    private init(onCancel: @escaping () -> Void) {
        let hosting = NSHostingController(rootView: OCRProgressView(model: model, onCancel: onCancel))
        hosting.sizingOptions = [.preferredContentSize]   // window tracks SwiftUI's fitting size
        let window = NSPanel(contentRect: NSRect(x: 0, y: 0, width: 420, height: 140),
                             styleMask: [.nonactivatingPanel, .fullSizeContentView],
                             backing: .buffered, defer: false)
        window.level = .floating
        window.backgroundColor = .clear
        window.isOpaque = false
        window.hasShadow = true
        window.isMovableByWindowBackground = false
        window.contentViewController = hosting
        if let screen = NSScreen.main {
            let frame = window.frame
            window.setFrameOrigin(NSPoint(x: screen.frame.midX - frame.width / 2, y: screen.frame.minY + 80))
        }
        super.init(window: window)
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }
}

@MainActor
final class OCRProgressModel: ObservableObject {
    @Published var info: OCRProgress = .preparing
}

private struct OCRProgressView: View {
    @ObservedObject var model: OCRProgressModel
    let onCancel: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text(model.info.label).font(.title3).fontWeight(.medium)
                Spacer()
                if let fraction = model.info.fraction {
                    Text("\(Int(fraction * 100))%")
                        .font(.title3.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
            // nil fraction -> indeterminate spinner bar; otherwise a filled bar.
            if let fraction = model.info.fraction {
                ProgressView(value: fraction)
            } else {
                ProgressView().progressViewStyle(.linear)
            }
            HStack {
                Spacer()
                Button("Cancel", action: onCancel)
                    .keyboardShortcut(.cancelAction)
            }
        }
        .frame(minWidth: 360)
        .padding(24)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
        .padding(10)
    }
}
