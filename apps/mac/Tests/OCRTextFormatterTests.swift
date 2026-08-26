import XCTest

final class OCRTextFormatterTests: XCTestCase {
    func testPreservingKeepsOriginalNewlines() {
        let text = "The quick brown fox\njumps over 13 lazy dogs"
        XCTAssertEqual(
            OCRTextFormatter.apply(text, preserveLinebreaks: true),
            text
        )
    }

    func testCollapsingJoinsLinesWithASingleSpace() {
        let text = "The quick brown fox\njumps over 13 lazy dogs"
        XCTAssertEqual(
            OCRTextFormatter.apply(text, preserveLinebreaks: false),
            "The quick brown fox jumps over 13 lazy dogs"
        )
    }

    func testCollapsingDropsBlankLines() {
        let text = "line one\n\nline two"
        XCTAssertEqual(
            OCRTextFormatter.apply(text, preserveLinebreaks: false),
            "line one line two"
        )
    }

    func testCollapsingTrimsWhitespaceAroundLinebreaks() {
        let text = "foo  \n  bar"
        XCTAssertEqual(
            OCRTextFormatter.apply(text, preserveLinebreaks: false),
            "foo bar"
        )
    }

    func testCollapsingHandlesKhmerMultiline() {
        let text = "ទម្រង់សម្រាប់\nពិនិត្យអក្ខរាវិរុទ្ធ"
        XCTAssertEqual(
            OCRTextFormatter.apply(text, preserveLinebreaks: false),
            "ទម្រង់សម្រាប់ ពិនិត្យអក្ខរាវិរុទ្ធ"
        )
    }

    func testSingleLineIsUnchangedWhenCollapsing() {
        let text = "just one line"
        XCTAssertEqual(
            OCRTextFormatter.apply(text, preserveLinebreaks: false),
            text
        )
    }
}
