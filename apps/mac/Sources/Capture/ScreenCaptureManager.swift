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

        let filter = SCContentFilter(display: display, excludingWindows: [])
        let config = SCStreamConfiguration()
        config.width = Int(display.width)
        config.height = Int(display.height)
        config.showsCursor = false

        let fullImage = try await SCScreenshotManager.captureImage(contentFilter: filter, configuration: config)

        // Convert selection rect (AppKit bottom-left origin, points) into the
        // captured image's top-left origin pixel space.
        let scale = CGFloat(fullImage.width) / screen.frame.width
        let flippedY = screen.frame.height - rect.origin.y - rect.height
        let cropRect = CGRect(
            x: rect.origin.x * scale,
            y: flippedY * scale,
            width: rect.width * scale,
            height: rect.height * scale
        )

        guard let cropped = fullImage.cropping(to: cropRect) else {
            throw ScreenCaptureError.captureFailed
        }
        return cropped
    }
}
