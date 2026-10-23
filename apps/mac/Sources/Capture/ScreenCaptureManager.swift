import ScreenCaptureKit
import AppKit

enum ScreenCaptureError: Error {
    case noDisplayFound
    case captureFailed
}

/// Captures a single still image of a screen region using ScreenCaptureKit,
/// the modern replacement for the deprecated CGWindowListCreateImage APIs.
enum ScreenCaptureManager {
    static func captureImage(of rect: CGRect, screen: NSScreen) async throws -> CGImage {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)

        let screenNumber = (screen.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? NSNumber)?.uint32Value
        guard let display = content.displays.first(where: { $0.displayID == screenNumber }) ?? content.displays.first else {
            throw ScreenCaptureError.noDisplayFound
        }

        // Capture only the selection, at the screen's native pixel density.
        // SCDisplay.width is in points, so sizing the stream from it captured
        // Retina screens at 1x: half the resolution, and on Khmer screen clips
        // 14% more character errors than the same clips at 2x (paired, n=225).
        let flippedY = screen.frame.height - rect.origin.y - rect.height
        let scale = screen.backingScaleFactor
        let config = SCStreamConfiguration()
        config.sourceRect = CGRect(x: rect.origin.x, y: flippedY, width: rect.width, height: rect.height)
        config.width = max(1, Int((rect.width * scale).rounded()))
        config.height = max(1, Int((rect.height * scale).rounded()))
        config.showsCursor = false

        let filter = SCContentFilter(display: display, excludingWindows: [])
        return try await SCScreenshotManager.captureImage(contentFilter: filter, configuration: config)
    }
}
