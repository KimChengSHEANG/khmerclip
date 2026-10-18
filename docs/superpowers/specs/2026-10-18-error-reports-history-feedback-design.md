# Error reports, History with images, and Feedback

Date: 2026-10-18 · Status: awaiting approval

## Why

KhmerClip users are everyday people with no GitHub account. When a capture
comes out wrong, or the app crashes, nothing reaches the developer today:
"Flag as Wrong" only copies the image/text pair into a local `feedback/`
folder on the user's own machine. The developer wants wrong captures,
crashes, and suggestions to land somewhere they already manage (GitHub
issues), with email (cheng900@gmail.com) as the fallback — and wants the
wrong-capture reports to double as correction data for training.

## Goals

1. **Crash reports are sent automatically** — no user action.
2. **History shows each scan's captured image** (layout A, below), and each
   row has a **Report** button that sends the image + recognized text (+ an
   optional correction) after the user sees exactly what will be sent.
3. **Send Feedback…** lets users send free-text suggestions.
4. All three reach the developer as **GitHub issues**; if GitHub fails, as an
   **email to cheng900@gmail.com**.
5. Works in **both apps**: the native Mac app (`apps/mac`, Swift/SwiftUI) and
   the Windows/Linux app (`apps/desktop`, Python/PySide6).

## Non-goals

- User accounts, sign-in, or any GitHub knowledge on the user's side.
- Replies back to the user inside the app.
- A dashboard or triage tooling beyond GitHub issues.
- Sending capture images automatically — only on the user's explicit Send.

## Anti-goals (failure even if it "works")

- A user's screen content becoming public without them knowingly sending it.
- Crash reports containing screenshots or the user's scanned text.
- A stranger able to spam the developer's repo/inbox through the endpoint.
- A crash loop producing hundreds of duplicate issues.
- Users being asked to do anything technical (tokens, accounts, browser forms).

## Decided by the developer

- Users have no GitHub accounts → reports go through a relay the developer owns.
- Crash reports: automatic. Capture reports: from the History page, user-initiated.
- History layout **A**: today's list, plus a thumbnail beside each scan and a
  Report button on every row; Copy / Delete / Clear All stay.
- "Keep capture images" becomes **on by default** in both apps
  (`apps/desktop/src/khmerclip/prefs.py` `keep_captures`,
  `apps/mac/Sources/Preferences/AppPreferences.swift` `keepCaptureImages`).
  Images stay local; they leave only when the user presses Send.
- Both apps get the features.

## Design

### Report window (both apps, from History → Report)

Shows: the captured image, "KhmerClip read: <text>", an optional
"What it should say" field (Khmer font), the line "This image and text will be
sent to the KhmerClip developer.", Cancel / Send. On success the row shows
"Reported"; on failure, a plain message and the report stays retryable.

Report payload: image (PNG), recognized text, correction (may be empty), the
existing `captures/<id>.json` OCR details, app version, OS, engine.

Report **replaces "Flag as Wrong"** in both History UIs. Keep writing the pair
into `feedback/` on Send too, so the existing eval tooling
(`training/eval/scripts/draft_real_fixtures.py`) keeps working; store the
correction in the pair's JSON. *(Builder's call whether the "Feedback Folder"
button stays — my call: hide it; it's a developer feature.)*

Rows without a kept image (older scans, or the pref turned off) show no
thumbnail and no Report button — same rule as today's flag button.

### Crash reports (my call on the mechanics)

- **Capture at crash time, send on next launch.** A crashing process can't
  send reliably. Write a crash file into the data dir; on next start, send
  pending crash files in the background and delete them on success.
  - Desktop: `sys.excepthook` + `threading.excepthook` (and Qt slot
    exceptions) write a traceback; `faulthandler` to a file for native crashes.
  - Mac: `NSSetUncaughtExceptionHandler` + signal handlers for the common fatal
    signals write a minimal record; Swift runtime traps are best-effort —
    acceptable ceiling. Builder may pick an equivalent approach (e.g. MetricKit
    diagnostics) if simpler.
- Contents: error/traceback, app version, OS + version, OCR engine, timestamp.
  **Never** image or scanned text.
- **Dedup:** the relay fingerprints (top frames + version) and comments
  "+1 (count)" on an existing open issue instead of opening a new one.
- Preferences gets **"Send crash reports"**, on by default.

### Send Feedback…

Menu item next to "History…" in the tray (`apps/desktop/src/khmerclip/ui/app.py`)
and menu bar (`apps/mac/Sources/App/AppDelegate.swift`). Window: multiline
text, optional "Your email (if you'd like a reply)", Send. Includes app
version + OS.

### Relay service (my call)

A small **Cloudflare Worker** (free tier), new directory `apps/report-relay/`,
deployed once by the developer with `wrangler`.

- `POST /report` with JSON `{type: "crash"|"capture"|"feedback", ...}`; image
  base64, capped (e.g. 2 MB image, 64 KB text).
- Holds a fine-grained GitHub token (issues + contents write on the reports
  repo only) as a Worker secret. Never in the app binary.
- Creates an issue labeled `crash` / `capture` / `feedback`. Images: GitHub's
  API has no issue-attachment upload, so commit the PNG to the reports repo
  (e.g. `reports/<id>.png`) and link/embed it in the issue body. Capture issues
  show recognized text and correction side by side.
- Fallback: if the GitHub call fails, email the same content (image attached)
  to cheng900@gmail.com via an email API (e.g. Resend free tier; builder's
  pick). If both fail, return an error so the app keeps the report and retries.
- Abuse limits: per-IP rate limit (Cloudflare rate-limiting binding or KV),
  size caps, `type` whitelist, plus a static app key header to drop casual
  noise (it's not a secret — it ships in the app; it only filters bots).
- Endpoint URL is a constant in each app.

### Offline / retry

Unsent reports (any type) are queued as files in the data dir and retried on
next launch and on next send. *(My call.)*

### Calls made after review (mine)

- **Training data reaches the developer via the reports repo.** For capture
  reports the relay commits a JSON sidecar next to the PNG
  (`reports/<id>.json`: recognized text, correction, OCR details, version,
  engine). A script to pull these into `fixtures_real/` is a follow-up, not
  this work. The local `feedback/` write stays only for the developer's own
  dogfooding.
- **Keep-images default flips for new installs only.** No migration overrides a
  stored "off" (desktop `Prefs.save()` writes every field; Mac UserDefaults
  likewise), out of respect for the privacy setting.
- **No first-run crash notice.** Instead, both apps replace the user's home
  directory in tracebacks/paths with `~` before writing the crash file.
- **Ownership and order.** The developer owns the Cloudflare account and the
  GitHub token. The relay ships and is deployed first; apps reference its URL
  as a constant set before release. Email fallback: Resend can send to the
  account owner's own address without a verified domain, which is enough here.
  If not, any transactional provider works.
- **Deletion.** Reports live in a private repo, so removal on request is a
  manual issue delete plus file delete. Git history keeps the blob; that's an
  acceptable ceiling for a private repo.

### Where reports land (developer's decision)

A new **private** repo, `KimChengSHEANG/khmerclip-reports`, holds all report
issues and committed images. Not the public `khmerclip` repo. The GitHub token
is scoped to that repo only. Issue bodies link the committed image by repo
path; it renders for the developer, who has access.

## Left to the builder

- Exact UI widgets/spacing within layout A; thumbnail size; Khmer font use
  follows existing `khmer_font` / SwiftUI defaults.
- Fingerprint algorithm, rate-limit numbers, email provider.
- Whether to share a JSON schema file between the apps and the Worker.
- Tests: Worker unit tests for routing/fallback/limits; app tests for queueing,
  crash-file write/read, and History rows with/without images, following the
  existing `apps/desktop/tests` and `apps/mac/Tests` style.
