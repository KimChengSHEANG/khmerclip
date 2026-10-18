import XCTest

final class KeepCapturesDefaultTests: XCTestCase {
    func testOnForNewInstallsOnly() {
        XCTAssertTrue(KeepCapturesDefault.resolve(stored: nil, hasHistory: false, completedOnboarding: false))
        XCTAssertFalse(KeepCapturesDefault.resolve(stored: nil, hasHistory: true, completedOnboarding: false))
        XCTAssertFalse(KeepCapturesDefault.resolve(stored: nil, hasHistory: false, completedOnboarding: true))
    }

    func testStoredChoiceWins() {
        XCTAssertTrue(KeepCapturesDefault.resolve(stored: true, hasHistory: true, completedOnboarding: true))
        XCTAssertFalse(KeepCapturesDefault.resolve(stored: false, hasHistory: false, completedOnboarding: false))
    }
}
