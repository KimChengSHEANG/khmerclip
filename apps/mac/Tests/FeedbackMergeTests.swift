import XCTest

final class FeedbackMergeTests: XCTestCase {
    func testMergeFeedbackAddsCorrection() throws {
        let input = Data(#"{"ocr_text":"ក"}"#.utf8)
        let out = mergeFeedback(input, correction: "ខ")
        let obj = try XCTUnwrap(JSONSerialization.jsonObject(with: out) as? [String: Any])
        XCTAssertEqual(obj["correction"] as? String, "ខ")
        XCTAssertEqual(obj["ocr_text"] as? String, "ក")
    }

    func testMergeFeedbackToleratesUnreadableInput() throws {
        let out = mergeFeedback(Data(), correction: "ខ")
        let obj = try XCTUnwrap(JSONSerialization.jsonObject(with: out) as? [String: Any])
        XCTAssertEqual(obj["correction"] as? String, "ខ")
    }
}
