import AppKit

enum CaptureMode {
    case text
    case barcode
}

/// Orchestrates the full snip flow: overlay selection -> screenshot -> OCR/barcode
/// -> clipboard -> history -> HUD. One instance is owned by AppDelegate.
@MainActor
final class CaptureCoordinator {
    private var overlayWindows: [NSWindow] = []
    private let preferences: AppPreferences
    private let historyStore: ScanHistoryStore
    private var hudWindowController: HUDWindowController?

    init(preferences: AppPreferences, historyStore: ScanHistoryStore) {
        self.preferences = preferences
        self.historyStore = historyStore
    }

    func beginCapture(mode: CaptureMode) {
        guard overlayWindows.isEmpty else { return }
        for screen in NSScreen.screens {
            let window = SelectionOverlayWindow(screen: screen)
            let view = SelectionOverlayView(frame: screen.frame)
            view.onComplete = { [weak self] rect in
                self?.finishSelection(rect: rect, screen: screen, mode: mode)
            }
            window.contentView = view
            window.makeKeyAndOrderFront(nil)
            overlayWindows.append(window)
        }
        NSApp.activate(ignoringOtherApps: true)
        NSCursor.hide()
    }

    private func finishSelection(rect: CGRect?, screen: NSScreen, mode: CaptureMode) {
        NSCursor.unhide()
        overlayWindows.forEach { $0.orderOut(nil) }
        overlayWindows.removeAll()

        guard let rect else { return }
        if preferences.playCaptureSound {
            NSSound(named: "Grab")?.play()
        }

        Task { [weak self] in
            guard let self else { return }
            do {
                let image = try await ScreenCaptureManager.captureImage(of: rect, screen: screen)
                try await self.recognize(image: image, mode: mode)
            } catch {
                await self.handleFailure(error)
            }
        }
    }

    /// Runs OCR/barcode detection on an already-in-hand image (a screen capture,
    /// or a file dropped/opened via the Image OCR window) and reports the result.
    func recognize(image: CGImage, mode: CaptureMode) async throws {
        switch mode {
        case .text:
            let text = try await TesseractEngine.recognizeText(in: image)
            await handleResult(text: text, kind: .text)
        case .barcode:
            let payloads = try await BarcodeEngine.detectPayloads(in: image)
            await handleResult(text: payloads.joined(separator: "\n"), kind: .barcode)
        }
    }

    func reportFailure(_ error: Error) async {
        await handleFailure(error)
    }

    private func handleResult(text: String, kind: ScanRecord.Kind) async {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            hudWindowController = HUDWindowController.showMessage("No text found")
            return
        }
        if preferences.autoCopyToClipboard {
            ClipboardManager.copy(text)
        }
        historyStore.add(ScanRecord(text: text, kind: kind))
        if preferences.showResultHUD {
            hudWindowController = HUDWindowController.showResult(text: text)
        }
    }

    private func handleFailure(_ error: Error) async {
        hudWindowController = HUDWindowController.showMessage("Capture failed: \(error.localizedDescription)")
    }
}
