import SwiftUI
import UniformTypeIdentifiers

/// "Drag an image here" utility window — TextSniper's other entry point besides
/// live screen selection: OCR an existing image file instead of the screen.
enum ImageOCRWindowPresenter {
    private static var controller: NSWindowController?

    static func show(coordinator: CaptureCoordinator) {
        if let controller {
            controller.showWindow(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }
        let hosting = NSHostingController(rootView: ImageOCRView(coordinator: coordinator))
        let window = NSWindow(contentViewController: hosting)
        window.title = "OCR Image File"
        window.styleMask = [.titled, .closable, .miniaturizable]
        window.center()
        let windowController = NSWindowController(window: window)
        controller = windowController
        windowController.showWindow(nil)
        NSApp.activate(ignoringOtherApps: true)
    }
}

struct ImageOCRView: View {
    let coordinator: CaptureCoordinator

    @State private var isTargeted = false
    @State private var statusText = "Drop an image, or choose a file"
    @State private var isProcessing = false

    var body: some View {
        VStack(spacing: 16) {
            ZStack {
                RoundedRectangle(cornerRadius: 12)
                    .strokeBorder(style: StrokeStyle(lineWidth: 2, dash: [6]))
                    .foregroundStyle(isTargeted ? Color.accentColor : .secondary)
                VStack(spacing: 8) {
                    Image(systemName: isProcessing ? "hourglass" : "photo.badge.plus")
                        .font(.system(size: 32))
                    Text(statusText)
                        .font(.callout)
                        .multilineTextAlignment(.center)
                        .foregroundStyle(.secondary)
                }
                .padding()
            }
            .frame(width: 320, height: 200)
            .onDrop(of: [.fileURL], isTargeted: $isTargeted) { providers in
                handleDrop(providers)
            }

            Button("Choose File…") { chooseFile() }
        }
        .padding(24)
    }

    private func handleDrop(_ providers: [NSItemProvider]) -> Bool {
        guard let provider = providers.first,
              provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) else { return false }

        provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier) { item, _ in
            guard let data = item as? Data,
                  let url = URL(dataRepresentation: data, relativeTo: nil) else { return }
            Task { await process(url: url) }
        }
        return true
    }

    private func chooseFile() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        if panel.runModal() == .OK, let url = panel.url {
            Task { await process(url: url) }
        }
    }

    @MainActor
    private func process(url: URL) async {
        guard let nsImage = NSImage(contentsOf: url) else {
            statusText = "Couldn't read that file as an image"
            return
        }
        await process(nsImage: nsImage)
    }

    @MainActor
    private func process(nsImage: NSImage) async {
        guard let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            statusText = "Couldn't decode that image"
            return
        }
        isProcessing = true
        statusText = "Recognizing text…"
        defer { isProcessing = false }
        do {
            try await coordinator.recognize(image: cgImage, mode: .text)
            statusText = "Done — check the result popup"
        } catch {
            statusText = "Failed: \(error.localizedDescription)"
            await coordinator.reportFailure(error)
        }
    }
}
