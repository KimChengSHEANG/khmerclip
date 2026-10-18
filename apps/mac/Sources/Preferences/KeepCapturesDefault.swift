/// "Keep capture images" defaults on for new installs only. An install that has run before
/// (it has history, or finished onboarding) and never chose keeps the old default: off.
enum KeepCapturesDefault {
    static func resolve(stored: Bool?, hasHistory: Bool, completedOnboarding: Bool) -> Bool {
        stored ?? !(hasHistory || completedOnboarding)
    }
}
