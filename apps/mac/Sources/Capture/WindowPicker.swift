import AppKit
import CoreGraphics

struct PickableWindow {
    let bounds: CGRect // global, top-left-origin screen coordinates (CGWindowListCopyWindowInfo convention)
    let ownerName: String
}

/// Backs the "click a window instead of dragging a region" mode (the same
/// space-bar toggle macOS's own screenshot tool uses), listing on-screen
/// windows via the classic Quartz Window Services API.
enum WindowPicker {
    static func onScreenWindows() -> [PickableWindow] {
        let options: CGWindowListOption = [.optionOnScreenOnly, .excludeDesktopElements]
        guard let list = CGWindowListCopyWindowInfo(options, kCGNullWindowID) as? [[String: AnyObject]] else {
            return []
        }
        let ownPID = Int32(ProcessInfo.processInfo.processIdentifier)

        return list.compactMap { info in
            guard let layer = info[kCGWindowLayer as String] as? Int, layer == 0 else { return nil }
            guard let ownerPID = info[kCGWindowOwnerPID as String] as? Int32, ownerPID != ownPID else { return nil }
            guard let boundsDict = info[kCGWindowBounds as String] as? [String: CGFloat] else { return nil }
            guard let x = boundsDict["X"], let y = boundsDict["Y"],
                  let width = boundsDict["Width"], let height = boundsDict["Height"],
                  width > 20, height > 20 else { return nil }
            let ownerName = info[kCGWindowOwnerName as String] as? String ?? ""
            return PickableWindow(bounds: CGRect(x: x, y: y, width: width, height: height), ownerName: ownerName)
        }
    }

    /// Finds the frontmost (first, since the list is already in front-to-back order)
    /// window whose bounds contain the given point, both in global top-left coordinates.
    static func window(at globalPoint: CGPoint, in windows: [PickableWindow]) -> PickableWindow? {
        windows.first { $0.bounds.contains(globalPoint) }
    }

    /// Converts CGWindowListCopyWindowInfo's global top-left-origin bounds into
    /// AppKit's bottom-left-origin coordinate space for a given screen.
    static func appKitRect(from globalBounds: CGRect, primaryScreenHeight: CGFloat) -> CGRect {
        CGRect(
            x: globalBounds.origin.x,
            y: primaryScreenHeight - globalBounds.origin.y - globalBounds.height,
            width: globalBounds.width,
            height: globalBounds.height
        )
    }
}
