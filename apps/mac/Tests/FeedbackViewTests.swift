import XCTest

final class FeedbackViewTests: XCTestCase {
    func testCanSendRequiresText() {
        XCTAssertFalse(FeedbackView.canSend(""))
        XCTAssertFalse(FeedbackView.canSend("  \n"))
        XCTAssertTrue(FeedbackView.canSend("សូមបន្ថែម"))
    }
}
