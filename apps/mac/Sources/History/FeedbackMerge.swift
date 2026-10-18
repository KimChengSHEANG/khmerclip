import Foundation

/// Returns the capture's meta JSON with `correction` added (an unreadable input yields just the correction).
func mergeFeedback(_ data: Data, correction: String) -> Data {
    var obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
    obj["correction"] = correction
    return (try? JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted, .sortedKeys])) ?? data
}
