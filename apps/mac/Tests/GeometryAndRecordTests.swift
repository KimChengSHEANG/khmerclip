import XCTest
// WindowPicker.swift and ScanRecord.swift are compiled directly into this test
// bundle (see project.yml) so the suite runs as a pure logic test with no app host.
import AppKit

final class WindowPickerTests: XCTestCase {
    // Front-to-back order: window(at:) must return the first window that contains
    // the point, even when a window behind it also contains it.
    func testHitTestReturnsFrontmost() {
        let front = PickableWindow(bounds: CGRect(x: 0, y: 0, width: 100, height: 100), ownerName: "Front")
        let back = PickableWindow(bounds: CGRect(x: 0, y: 0, width: 200, height: 200), ownerName: "Back")
        let hit = WindowPicker.window(at: CGPoint(x: 50, y: 50), in: [front, back])
        XCTAssertEqual(hit?.ownerName, "Front")
    }

    func testHitTestFallsThroughToContainingWindow() {
        let a = PickableWindow(bounds: CGRect(x: 0, y: 0, width: 10, height: 10), ownerName: "A")
        let b = PickableWindow(bounds: CGRect(x: 100, y: 100, width: 50, height: 50), ownerName: "B")
        let hit = WindowPicker.window(at: CGPoint(x: 120, y: 120), in: [a, b])
        XCTAssertEqual(hit?.ownerName, "B")
    }

    func testHitTestMissReturnsNil() {
        let a = PickableWindow(bounds: CGRect(x: 0, y: 0, width: 10, height: 10), ownerName: "A")
        XCTAssertNil(WindowPicker.window(at: CGPoint(x: 500, y: 500), in: [a]))
    }

    // Top-left-origin global bounds -> AppKit bottom-left-origin. A window at the
    // top of a 900pt-tall screen must map to a high y; one at the bottom to y≈0.
    func testAppKitRectFlipsY() {
        let topWindow = CGRect(x: 30, y: 0, width: 400, height: 300)
        let converted = WindowPicker.appKitRect(from: topWindow, primaryScreenHeight: 900)
        XCTAssertEqual(converted, CGRect(x: 30, y: 600, width: 400, height: 300))

        let bottomWindow = CGRect(x: 0, y: 600, width: 400, height: 300)
        let convertedBottom = WindowPicker.appKitRect(from: bottomWindow, primaryScreenHeight: 900)
        XCTAssertEqual(convertedBottom, CGRect(x: 0, y: 0, width: 400, height: 300))
    }
}

final class ScanRecordTests: XCTestCase {
    func testDefaultKindIsText() {
        XCTAssertEqual(ScanRecord(text: "hi").kind, .text)
    }

    func testEachRecordGetsUniqueID() {
        XCTAssertNotEqual(ScanRecord(text: "a").id, ScanRecord(text: "a").id)
    }

    func testCodableRoundTrip() throws {
        let original = ScanRecord(text: "កម្ពុជា mix 123", kind: .barcode, date: Date(timeIntervalSince1970: 1_700_000_000))
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(ScanRecord.self, from: data)
        XCTAssertEqual(decoded, original)
    }
}
