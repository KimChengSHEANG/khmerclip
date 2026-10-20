# Error Reports, History with Images, and Feedback — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Crash reports (automatic), wrong-capture reports (from History, with image), and feedback reach the developer as GitHub issues in a private repo, falling back to email, from both the Mac and Windows/Linux apps.

**Architecture:** A Cloudflare Worker (`apps/report-relay`) is the only thing holding credentials; it turns a JSON report into a GitHub issue (image + JSON sidecar committed to the reports repo) or, failing that, a Resend email. Each app writes every report as a JSON file into an `outbox/` folder in its data dir and flushes that folder in the background (on launch and right after each Send). Crash handlers write straight into the outbox or a pending file; nothing is sent from a crashing process.

**Tech Stack:** Worker: plain ES-module JavaScript, `node:test`, `wrangler` (deploy only, via `npx`). Desktop: Python 3.12, PySide6, `urllib` (no new deps). Mac: Swift 5, SwiftUI, `URLSession`, XCTest.

**Spec:** `docs/superpowers/specs/2026-10-18-error-reports-history-feedback-design.md`

## Global Constraints

- Reports repo: `KimChengSHEANG/khmerclipper-reports` (private). Never the public `khmerclipper` repo.
- Fallback email: `cheng900@gmail.com`.
- GitHub token lives only as a Worker secret, scoped to the reports repo (Issues + Contents: read/write). Never in an app.
- Labels: `crash`, `capture`, `feedback`.
- Limits (relay rejects with 413; apps pre-shrink images): PNG ≤ 2,000,000 bytes decoded; each text field ≤ 65,536 chars; email ≤ 254 chars; request body ≤ 3,000,000 bytes.
- Header `X-KhmerClipper-Key: kc-reports-1` on every report (not a secret; filters bots only).
- Report URL constant in each app starts as `https://khmerclipper-report-relay.REPLACE_ME.workers.dev/report`. While a URL contains `REPLACE_ME`, apps queue but never send. Task 10 sets the real URL.
- Crash reports never contain an image or scanned text. The home directory is replaced with `~` in all crash text.
- "Keep capture images" default → on (new installs only; no migration). New pref "Send crash reports automatically", default on.
- History layout A: thumbnail + text + Report button per row; Copy / Delete / Clear All remain.
- No new runtime dependencies in either app.

### Wire format (shared by all tasks)

`POST` JSON, one object:

| field | type | crash | capture | feedback |
|---|---|---|---|---|
| `type` | `"crash"` \| `"capture"` \| `"feedback"` | ✓ | ✓ | ✓ |
| `id` | 32 lowercase hex chars (client UUID, no dashes) | ✓ | ✓ | ✓ |
| `app_version`, `os`, `engine` | string | ✓ | ✓ | ✓ |
| `signature` | string, e.g. `"ValueError at ocr/trocr.py:decode"` | ✓ | | |
| `details` | string (traceback / stack) | ✓ | | |
| `image_png_b64` | base64 PNG | | ✓ | |
| `ocr_text`, `correction` | string (`correction` may be `""`) | | ✓ | |
| `meta` | object (the capture's `<id>.json`) | | ✓ | |
| `message` | string | | | ✓ |
| `email` | string, may be `""` | | | ✓ |

Responses: `200 {"ok":true,"via":"github"|"email"}` → client deletes the outbox file. `400/401/413` → client deletes (never retry a rejected report). Anything else (429, 5xx, network error) → keep and retry later.

## Review Focus

1. **Huge full-screen Retina capture** (PNG > 2 MB) → the app downscales before queueing; Report never silently fails. *(Task 2 `test_capture_payload_downscales_large_png`, Task 6 `testCapturePayloadDownscalesLargeImage`.)*
2. **Offline at Send time** → report stays in `outbox/`, the row still shows "Reported", and it's sent on next launch. *(Task 2 `test_flush_keeps_file_on_network_error`, Task 6 `testFlushKeepsFileOnServerError`.)*
3. **Crash loop** (same crash every launch) → one issue with "+1" comments, not N issues. *(Task 1 `crash with known fingerprint comments instead of opening`.)*
4. **User text containing markdown, `~~~` fences, or @mentions** → the issue body can't ping people or break formatting. *(Task 1 `user text is fenced and cannot break out`.)*
5. **Relay URL not configured yet** (`REPLACE_ME`) → nothing is sent, nothing is lost. *(Task 2 `test_flush_does_nothing_until_url_configured`, Task 6 `testFlushSkipsUnconfiguredURL`.)*

---

### Task 1: Report relay Worker

**Files:**
- Create: `apps/report-relay/src/index.js`
- Create: `apps/report-relay/test/index.test.js`
- Create: `apps/report-relay/package.json` (`{"type":"module","private":true,"scripts":{"test":"node --test"}}`)
- Create: `apps/report-relay/wrangler.toml`
- Create: `apps/report-relay/README.md` (deploy checklist — content in Task 10)
- Modify: `Makefile` (add `check-relay: cd apps/report-relay && node --test`, add it to `check` and `.PHONY`)

**Interfaces:**
- Produces: `export async function handle(request, env, fetchFn)` → `Response`; `export default { fetch: (req, env) => handle(req, env, fetch) }`.
- `env`: `GITHUB_TOKEN`, `RESEND_API_KEY` (secrets); `APP_KEY="kc-reports-1"`, `REPORTS_REPO="KimChengSHEANG/khmerclipper-reports"`, `EMAIL_TO="cheng900@gmail.com"` (vars); `RATE_LIMITER` (rate-limit binding, 10 requests / 60 s, keyed by `CF-Connecting-IP`).
- `wrangler.toml`: `name = "khmerclipper-report-relay"`, `main = "src/index.js"`, `compatibility_date = "2026-10-01"`, `[vars]` above, a ratelimit binding named `RATE_LIMITER` with `limit = 10, period = 60` (use the binding syntax current `wrangler` documents).

Behavior:
- Only `POST /report`; otherwise 404. Wrong or missing `X-KhmerClipper-Key` → 401. `await env.RATE_LIMITER.limit({key: ip})` returns `{success:false}` → 429. Body or any field over its limit → 413. Unknown `type`, a missing required field for the type, or `id` not matching `/^[0-9a-f]{32}$/` → 400.
- GitHub (`https://api.github.com`; headers `Authorization: Bearer <token>`, `User-Agent: khmerclipper-report-relay`, `Accept: application/vnd.github+json`):
  - capture: `PUT /repos/{repo}/contents/reports/{id}.png` (base64 content), then `PUT …/contents/reports/{id}.json` (sidecar `{ocr_text, correction, meta, app_version, os, engine}`). A 422 on either means "already uploaded"; continue. Then `POST /repos/{repo}/issues`.
  - crash: `fp = sha256hex(app_version + "\n" + signature).slice(0, 16)`. `GET /search/issues?q=repo:{repo} is:issue is:open label:crash "fp:{fp}"` (URL-encoded). If `total_count > 0` → `POST …/issues/{items[0].number}/comments` with body `+1 · {app_version} · {os}`. Otherwise create the issue with `<!-- fp:{fp} -->` in its body. Search-index lag can let a near-simultaneous duplicate through; that's an accepted ceiling.
  - Titles (variable part single-line, truncated to 60 chars): `Wrong text: {ocr_text}`, `Crash: {signature}`, `Feedback: {message}`. Labels `[type]`.
  - Body: a line `App {app_version} · {os} · {engine}`. Every user-supplied string goes inside a fence of `~` characters one longer than the longest `~` run in that string (minimum 3). Capture bodies embed `![capture](https://github.com/{repo}/blob/main/reports/{id}.png?raw=true)` and show "KhmerClipper read" / "Should say" fences ("Should say" only when `correction` isn't empty). Feedback shows `Reply to:` plus a fenced email only when the email isn't empty. Crash bodies fence `details`.
- If any GitHub call (other than a tolerated 422) returns non-2xx or throws → `POST https://api.resend.com/emails` with `{from: "KhmerClipper <onboarding@resend.dev>", to: [EMAIL_TO], subject: title, text: body, attachments: capture ? [{filename: "{id}.png", content: image_png_b64}] : []}`. Resend 2xx → `200 via:"email"`; otherwise 502.

- [ ] **Step 1: Write the failing tests** in `test/index.test.js` using a fake `fetchFn` that records `(url, init)` and returns scripted `Response`s, and a fake `env` whose `RATE_LIMITER.limit` returns `{success:true}` unless the test overrides it:

```js
test('rejects wrong key')                         // 401; fetchFn never called
test('rejects bad id and unknown type')           // 400 for each
test('rejects oversized image')                   // image decoding to 2_000_001 bytes → 413
test('rate limited')                              // limit → {success:false} → 429
test('capture uploads png + sidecar then opens labeled issue')
  // call order: PUT …/contents/reports/<id>.png, PUT …/contents/reports/<id>.json, POST …/issues
  // issue JSON: labels deepEqual ['capture']; title startsWith 'Wrong text: '; body includes `reports/<id>.png?raw=true`
test('capture tolerates 422 on re-upload')        // both PUTs → 422; issue still created; 200 via 'github'
test('crash with known fingerprint comments instead of opening')
  // search → {total_count:1, items:[{number:7}]}; one POST to …/issues/7/comments; no POST to …/issues
test('new crash opens issue with fp marker')      // body matches /<!-- fp:[0-9a-f]{16} -->/; labels ['crash']
test('user text is fenced and cannot break out')
  // message 'hi ~~~\n@octocat **x**' → body includes '~~~~\nhi ~~~\n@octocat **x**\n~~~~'
test('github failure falls back to email with attachment')
  // POST …/issues → 500; Resend body: to deepEqual ['cheng900@gmail.com'], attachments[0].filename === '<id>.png'; 200 via 'email'
test('both fail → 502')
```

- [ ] **Step 2: Run to verify they fail** — `cd apps/report-relay && node --test` → FAIL (`handle` not exported).
- [ ] **Step 3: Implement `handle` in `src/index.js`** per Behavior. Use `crypto.subtle.digest('SHA-256', …)` for the fingerprint and compute decoded size from the base64 length (both work in Workers and Node 20+).
- [ ] **Step 4: Run to verify pass** — `node --test` → all pass.
- [ ] **Step 5: Commit** — `git add apps/report-relay Makefile && git commit -m "feat(relay): Cloudflare Worker that files reports as GitHub issues with email fallback"`

---

### Task 2: Desktop report outbox

**Files:**
- Create: `apps/desktop/src/khmerclipper/reports.py`
- Test: `apps/desktop/tests/test_reports.py`

**Interfaces:**
- Produces (module `khmerclipper.reports`):
  - `REPORT_URL = "https://khmerclipper-report-relay.REPLACE_ME.workers.dev/report"`, `APP_KEY = "kc-reports-1"`, `MAX_PNG_BYTES = 2_000_000`
  - `outbox_dir() -> Path` → `paths.data_dir() / "outbox"`
  - `base_fields(engine: str) -> dict` → `{"id": uuid4().hex, "app_version": __version__, "os": platform.platform(), "engine": engine}`
  - `scrub_home(text: str) -> str` — replaces `str(Path.home())` with `~`
  - `capture_payload(image_path: Path, meta: dict, ocr_text: str, correction: str, engine: str) -> dict`
  - `feedback_payload(message: str, email: str, engine: str) -> dict`
  - `crash_payload(signature: str, details: str, engine: str) -> dict` (both strings scrubbed)
  - `enqueue(payload: dict, outbox: Path | None = None) -> Path` — `paths.atomic_write_text` to `<outbox>/<id>.json`
  - `drop_type(kind: str, outbox: Path | None = None) -> None` — deletes queued files whose `type == kind`
  - `post_json(payload: dict, url: str = REPORT_URL, timeout: float = 20) -> int` — HTTP status; network failure → `0`
  - `flush(outbox: Path | None = None, post: Callable[[dict], int] | None = None, url: str = REPORT_URL) -> int` — number sent. Returns 0 at once if `"REPLACE_ME" in url`. `post` defaults to `lambda d: post_json(d, url)`. Deletes on 200/400/401/413, keeps otherwise. Deletes files that aren't valid JSON.

- [ ] **Step 1: Write the failing tests** (`URL = "https://relay.test/report"`, `UNSET = "https://khmerclipper-report-relay.REPLACE_ME.workers.dev/report"`):

```python
def test_enqueue_and_flush_deletes_on_success(tmp_path):
    p = reports.enqueue(reports.feedback_payload("សួស្តី", "", "tesseract"), tmp_path)
    sent = []
    assert reports.flush(tmp_path, post=lambda d: sent.append(d) or 200, url=URL) == 1
    assert not p.exists() and sent[0]["message"] == "សួស្តី" and sent[0]["type"] == "feedback"

def test_flush_keeps_file_on_network_error(tmp_path):
    p = reports.enqueue(reports.feedback_payload("x", "", "tesseract"), tmp_path)
    assert reports.flush(tmp_path, post=lambda d: 0, url=URL) == 0 and p.exists()
    reports.flush(tmp_path, post=lambda d: 503, url=URL)
    assert p.exists()

def test_flush_drops_rejected_reports(tmp_path):
    p = reports.enqueue(reports.feedback_payload("x", "", "tesseract"), tmp_path)
    reports.flush(tmp_path, post=lambda d: 413, url=URL)
    assert not p.exists()

def test_flush_does_nothing_until_url_configured(tmp_path):
    p = reports.enqueue(reports.feedback_payload("x", "", "tesseract"), tmp_path)
    called = []
    assert reports.flush(tmp_path, post=called.append, url=UNSET) == 0 and p.exists() and called == []

def test_drop_type_keeps_others(tmp_path):
    reports.enqueue(reports.crash_payload("E", "d", "tesseract"), tmp_path)
    keep = reports.enqueue(reports.feedback_payload("x", "", "tesseract"), tmp_path)
    reports.drop_type("crash", tmp_path)
    assert list(tmp_path.glob("*.json")) == [keep]

def test_crash_payload_scrubs_home():
    d = reports.crash_payload("E at a.py:f", f'File "{Path.home()}/x.py"', "trocr")
    assert str(Path.home()) not in d["details"] and '"~/x.py"' in d["details"]

def test_capture_payload_downscales_large_png(tmp_path):
    big = tmp_path / "big.png"
    Image.effect_noise((2400, 1600), 100).convert("RGB").save(big)   # well over 2 MB as PNG
    d = reports.capture_payload(big, {"ocr_text": "ក"}, "ក", "", "tesseract")
    png = base64.b64decode(d["image_png_b64"])
    assert len(png) <= reports.MAX_PNG_BYTES and png.startswith(b"\x89PNG")
```

- [ ] **Step 2: Run to verify fail** — `cd apps/desktop && uv run --extra dev pytest tests/test_reports.py -q` → FAIL (no module).
- [ ] **Step 3: Implement `khmerclipper/reports.py`.** Downscale: halve both dimensions with `Image.LANCZOS` until the PNG fits. `post_json` uses `urllib.request` like `updater.fetch_latest`, with headers `Content-Type: application/json`, `X-KhmerClipper-Key`, `User-Agent: KhmerClipper/<version>`. `HTTPError` → its `.code`; any other `OSError` → 0.
- [ ] **Step 4: Run to verify pass** — same command → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(desktop): report outbox with retrying flush"`

---

### Task 3: Desktop crash capture and prefs

**Files:**
- Create: `apps/desktop/src/khmerclipper/crash.py`
- Modify: `apps/desktop/src/khmerclipper/prefs.py` (`keep_captures: bool = True`; add `send_crash_reports: bool = True`)
- Modify: `apps/desktop/src/khmerclipper/ui/prefs_window.py` (`CHECKS` += `("send_crash_reports", "Send crash reports automatically")`)
- Modify: `apps/desktop/src/khmerclipper/ui/app.py` (`run_gui` installs, collects, and flushes; `Controller.handle_error` records unexpected errors)
- Test: `apps/desktop/tests/test_crash.py`. Update `tests/test_app.py::test_keep_captures_saves_image_with_ocr_details`: the default is now on, so set `controller.prefs.keep_captures = False` before the "off" assertion and change its comment.

**Interfaces:**
- Consumes: `reports.crash_payload`, `reports.enqueue`, `reports.flush`, `reports.drop_type`, `reports.outbox_dir`.
- Produces (module `khmerclipper.crash`):
  - `signature(exc: BaseException) -> str` → `"{type(exc).__name__} at {path}:{func}"` for the innermost traceback frame. `path` is relative to the `khmerclipper` package dir when inside it, otherwise the basename. With no traceback, it's just the type name.
  - `record(exc: BaseException, engine: str, outbox: Path | None = None) -> Path` — `details` = `"".join(traceback.format_exception(exc))`
  - `install(enabled: Callable[[], bool], engine: Callable[[], str], outbox: Path | None = None) -> None` — wraps `sys.excepthook` and `threading.excepthook`. Each records only when `enabled()`, then calls the previous hook; a failure while recording never masks the original. Enables `faulthandler` into `(outbox or reports.outbox_dir()).parent / "fault.log"`, keeping the file object referenced at module level.
  - `collect_fault_log(engine: str, outbox: Path | None = None) -> None` — if `fault.log` beside the outbox is non-empty, enqueue `crash_payload("Native crash", text, engine)` and truncate the file.
- `run_gui` wiring, right after `prefs = Prefs.load()`:
  1. `crash.install(lambda: prefs.send_crash_reports, lambda: prefs.engine)`
  2. If `send_crash_reports`, run `crash.collect_fault_log(prefs.engine)`. Otherwise, truncate the log and run `reports.drop_type("crash")`.
  3. After the controller exists, run `controller.submit(lambda p, c: reports.flush())`.
- `Controller.handle_error`: in the non-`KhmerClipperError` branch, call `crash.record(error, self.prefs.engine)` when `self.prefs.send_crash_reports`.

- [ ] **Step 1: Write the failing tests**

```python
def test_record_writes_scrubbed_crash_without_scanned_text(tmp_path):
    try:
        raise ValueError(f"bad {Path.home()}/secret.png")
    except ValueError as e:
        path = crash.record(e, "trocr", tmp_path)
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["type"] == "crash" and d["signature"].startswith("ValueError at ")
    assert str(Path.home()) not in d["details"] and "image_png_b64" not in d and "ocr_text" not in d

def test_excepthook_respects_pref_and_chains(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(sys, "excepthook", lambda *a: seen.append(a))
    monkeypatch.setattr(threading, "excepthook", lambda args: None)
    enabled = [False]
    crash.install(lambda: enabled[0], lambda: "tesseract", tmp_path / "outbox")
    err = RuntimeError("x")
    sys.excepthook(RuntimeError, err, None)
    assert list((tmp_path / "outbox").glob("*.json")) == [] and len(seen) == 1
    enabled[0] = True
    sys.excepthook(RuntimeError, err, None)
    assert len(list((tmp_path / "outbox").glob("*.json"))) == 1 and len(seen) == 2

def test_fault_log_becomes_one_crash(tmp_path):
    (tmp_path / "fault.log").write_text("Fatal Python error: Segmentation fault", encoding="utf-8")
    crash.collect_fault_log("tesseract", tmp_path / "outbox")
    [p] = (tmp_path / "outbox").glob("*.json")
    assert json.loads(p.read_text(encoding="utf-8"))["signature"] == "Native crash"
    assert (tmp_path / "fault.log").read_text(encoding="utf-8") == ""

def test_prefs_defaults():
    p = Prefs()
    assert p.keep_captures is True and p.send_crash_reports is True
```
  In `test_app.py`, add `test_unexpected_error_is_recorded(controller, monkeypatch)`: monkeypatch `app_module.crash.record` to append to a list. `controller.handle_error(ZeroDivisionError())` records once. `handle_error(KhmerClipperError("x"))` and `handle_error(Cancelled())` record nothing. With `send_crash_reports = False`, `ZeroDivisionError` records nothing.
  (`crash.install` replaces global hooks, so the test's `monkeypatch` restores them.)

- [ ] **Step 2: Run to verify fail** — `uv run --extra dev pytest tests/test_crash.py tests/test_app.py -q` → FAIL.
- [ ] **Step 3: Implement** `crash.py`, the prefs fields, the prefs window row, the `run_gui` wiring, and the `handle_error` hook.
- [ ] **Step 4: Run the whole desktop suite** — `make check-desktop` (from the repo root) → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(desktop): automatic crash reports and on-by-default capture images"`

---

### Task 4: Desktop History layout A and Report dialog

**Files:**
- Create: `apps/desktop/src/khmerclipper/ui/report_dialog.py`
- Modify: `apps/desktop/src/khmerclipper/ui/history_window.py`
- Modify: `apps/desktop/src/khmerclipper/history.py`: `flag(record_id, correction: str = "")` writes `correction` into the copied `feedback/<id>.json`; add `capture_meta(record_id) -> dict` (parsed `captures/<id>.json`, `{}` if missing or unreadable). The class docstring says `feedback/` now marks a scan as "Reported".
- Modify: `apps/desktop/src/khmerclipper/ui/app.py` (`TrayApp.show_history` passes a `report` callback)
- Test: `apps/desktop/tests/test_ui_widgets.py`, `apps/desktop/tests/test_history.py`

**Interfaces:**
- Consumes: `reports.capture_payload`, `reports.enqueue`, `reports.flush`; `HistoryStore.has_capture/is_flagged/flag/capture_meta/captures_dir`.
- Produces:
  - `ReportDialog(QDialog)(image_path: Path, ocr_text: str)` with attributes `correction: QPlainTextEdit` (prefilled with `ocr_text`, `khmer_font`) and `send_button`. `correction_text() -> str` returns `""` when the stripped text equals the stripped `ocr_text`, otherwise the stripped text. The dialog shows the image (scaled to at most 480 px wide), the label `KhmerClipper read`, the text, the label `What it should say`, the exact line `This image and text will be sent to the KhmerClipper developer.`, and Cancel / Send.
  - `HistoryWindow(store, copy, report: Callable[[ScanRecord, str], None] | None = None)`. Each list row is a widget set via `setItemWidget`: a 72×40 thumbnail `QLabel` (only when `has_capture`), the text preview, and a `QPushButton` with object name `report_button`. It reads `Report`, or reads `Reported` and is disabled when `is_flagged`. There's no button when there's no capture. Remove the "Flag as Wrong" and "Feedback Folder" buttons. Selection, Copy, Delete, Clear All, and double-click-to-copy keep working.
  - `HistoryWindow.report_record(record: ScanRecord) -> None` opens `ReportDialog`. When accepted: `store.flag(record.id, correction)`, then `report(record, correction)`, then `refresh()`.
  - `TrayApp._report(record, correction)`: `meta = history.capture_meta(record.id)`. Enqueue `reports.capture_payload(history.captures_dir / f"{record.id}.png", meta, meta.get("ocr_text", record.text), correction, prefs.engine)`, then `self.c.submit(lambda p, c: reports.flush(), on_done=…)`. The popup shows `Thanks — your report was sent.` when flush returned ≥ 1, otherwise `Report saved. It will be sent when you're online.`

- [ ] **Step 1: Write the failing tests**

```python
# test_history.py
def test_flag_stores_correction(tmp_path):
    store = HistoryStore(tmp_path / "h.json"); rec = ScanRecord("ក")
    store.add(rec, image=Image.new("RGB", (8, 4)), meta={"ocr_text": "ក"})
    store.flag(rec.id, "ខ")
    saved = json.loads((tmp_path / "feedback" / f"{rec.id}.json").read_text(encoding="utf-8"))
    assert saved == {"ocr_text": "ក", "correction": "ខ"}
    assert store.capture_meta(rec.id) == {"ocr_text": "ក"}

# test_ui_widgets.py
def test_history_rows_show_report_only_for_captures(qtbot, tmp_path):
    store = HistoryStore(tmp_path / "h.json")
    store.add(ScanRecord("plain")); rec = ScanRecord("ក")
    store.add(rec, image=Image.new("RGB", (8, 4)), meta={"ocr_text": "ក"})
    w = HistoryWindow(store, lambda t: None, report=lambda r, c: None); qtbot.addWidget(w); w.refresh()
    buttons = [w.list.itemWidget(w.list.item(i)).findChild(QPushButton, "report_button") for i in range(2)]
    assert buttons[0] is not None and buttons[0].text() == "Report" and buttons[1] is None

def test_report_flow_marks_reported(qtbot, tmp_path, monkeypatch):
    # same store as above; monkeypatch ReportDialog.exec → QDialog.DialogCode.Accepted
    # and ReportDialog.correction_text → "ខ"; reported = []
    # w.report_record(rec) → reported == [(rec.id, "ខ")], store.is_flagged(rec.id),
    # row 0 report_button text "Reported" and not enabled

def test_report_dialog_unchanged_text_means_no_correction(qtbot, tmp_path):
    png = tmp_path / "c.png"; Image.new("RGB", (8, 4)).save(png)
    d = ReportDialog(png, "ក"); qtbot.addWidget(d)
    assert d.correction_text() == ""
    d.correction.setPlainText("ខ")
    assert d.correction_text() == "ខ"
```
  `test_history_window_copy_delete_clear` must still pass unchanged.

- [ ] **Step 2: Run to verify fail** — `uv run --extra dev pytest tests/test_ui_widgets.py tests/test_history.py -q` → FAIL.
- [ ] **Step 3: Implement** the dialog, the row widgets, `flag(correction)`/`capture_meta`, and the TrayApp callback.
- [ ] **Step 4: Run** `make check-desktop` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(desktop): history thumbnails with Report button"`

---

### Task 5: Desktop Send Feedback

**Files:**
- Create: `apps/desktop/src/khmerclipper/ui/feedback_dialog.py`
- Modify: `apps/desktop/src/khmerclipper/ui/app.py` (menu item `Send Feedback…` right after `History…`; `TrayApp.show_feedback`)
- Test: `apps/desktop/tests/test_ui_widgets.py`. In `tests/test_app.py`, update the menu-text assertion to `["Capture Text", "OCR Image File…", "History…", "Send Feedback…", "Preferences…", "Check for Updates…", "Quit KhmerClipper"]`.

**Interfaces:**
- Produces: `FeedbackDialog(QDialog)()` with:
  - `message: QPlainTextEdit` (`khmer_font`)
  - `email: QLineEdit` (placeholder `Your email (if you'd like a reply)`)
  - `send_button`, enabled only while `message` has non-whitespace text
  - `values() -> tuple[str, str]` (both stripped)
- `TrayApp.show_feedback()`: when accepted, enqueue `reports.feedback_payload(msg, email, prefs.engine)`, flush in the background, and show the same two popup messages as Task 4.

- [ ] **Step 1: Failing test** — `test_feedback_dialog_requires_message(qtbot)`: send is disabled at first; still disabled after `message.setPlainText("  ")`; enabled after `"សូមបន្ថែម"`; with `email.setText(" a@b.c ")`, `values() == ("សូមបន្ថែម", "a@b.c")`.
- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `make check-desktop` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(desktop): Send Feedback menu item"`

---

### Task 6: Mac report core (payload + outbox)

**Files:**
- Create: `apps/mac/Sources/Reports/Report.swift`
- Create: `apps/mac/Sources/Reports/ReportOutbox.swift`
- Create: `apps/mac/Tests/ReportTests.swift`
- Modify: `apps/mac/project.yml` (add both new source files to `KhmerClipperTests.sources`)

**Interfaces:**
- Produces:
  - `enum Report` with static builders that return `[String: Any]` in the wire format:
    - `capture(pngURL: URL, meta: [String: Any], ocrText: String, correction: String, engine: String)`
    - `feedback(message: String, email: String, engine: String)`
    - `crash(signature: String, details: String, engine: String)`
    - Also `static func scrubHome(_ s: String) -> String` (replaces `NSHomeDirectory()` with `~`) and `static let maxPNGBytes = 2_000_000`.
    - `id` = `UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()`. `app_version` comes from `Bundle.main` `CFBundleShortVersionString` (fallback `"dev"`). `os` = `"macOS " + ProcessInfo.processInfo.operatingSystemVersionString`.
  - `struct ReportOutbox { let dir: URL }` with:
    - `static let unsetURL = URL(string: "https://khmerclipper-report-relay.REPLACE_ME.workers.dev/report")!`, `static var reportURL = unsetURL`, `static let appKey = "kc-reports-1"`
    - `func enqueue(_ payload: [String: Any]) throws -> URL`
    - `func dropType(_ type: String)`
    - `func flush(url: URL = ReportOutbox.reportURL, post: ([String: Any]) async -> Int) async -> Int` — same delete/keep rules as the wire format; returns 0 without calling `post` when `url.absoluteString.contains("REPLACE_ME")`
    - `static func post(_ payload: [String: Any], to url: URL) async -> Int`
  - `extension ReportOutbox { static let shared = ReportOutbox(dir: ScanHistoryStore.dataDir.appendingPathComponent("outbox", isDirectory: true)) }` lives at the bottom of `ScanHistoryStore.swift`, so the test target doesn't need that file.

- [ ] **Step 1: Failing tests** in `ReportTests.swift` (a fresh temp dir per test):
  - `testEnqueueAndFlushDeletesOnSuccess`: `post` returns 200 → flush returns 1, the file is gone, and the posted `type == "feedback"`. Pass `url: URL(string: "https://relay.test/report")!`.
  - `testFlushKeepsFileOnServerError`: 503, then 0 → file kept
  - `testFlushDropsRejected`: 413 → file gone
  - `testFlushSkipsUnconfiguredURL`: `url: ReportOutbox.unsetURL` → returns 0, `post` never called, file kept
  - `testCrashScrubsHome`: `NSHomeDirectory()` absent from `details`, `"~/"` present, no `image_png_b64` key
  - `testCapturePayloadDownscalesLargeImage`: write a 2400×1600 random-noise PNG (fill `CGContext` bytes with `UInt8.random`); decoded `image_png_b64` is ≤ `Report.maxPNGBytes` and starts with the PNG magic bytes
- [ ] **Step 2: Run to verify fail** — `make -C apps/mac test` → FAIL (missing types).
- [ ] **Step 3: Implement.** Downscale by halving with a `CGContext` redraw until the PNG data fits. `post` uses `URLSession.shared.data(for:)` with the same headers as desktop. A non-HTTP failure → 0.
- [ ] **Step 4: Run** `make -C apps/mac test` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(mac): report payloads and retrying outbox"`

---

### Task 7: Mac crash capture and prefs

**Files:**
- Create: `apps/mac/Sources/Reports/CrashReporter.swift`
- Modify: `apps/mac/Sources/Preferences/AppPreferences.swift` (`keepCaptureImages` default `true`; add `sendCrashReports` under key `"sendCrashReports"`, default `true`)
- Modify: `apps/mac/Sources/Preferences/PreferencesView.swift` (`Toggle("Send crash reports automatically", isOn: $preferences.sendCrashReports)` after the keep-images toggle)
- Modify: `apps/mac/Sources/App/AppDelegate.swift` (`applicationDidFinishLaunching`)
- Modify: `apps/mac/project.yml` (add `CrashReporter.swift` to test sources)
- Test: `apps/mac/Tests/ReportTests.swift`

**Interfaces:**
- Consumes: `Report.crash`, `ReportOutbox`.
- Produces: `enum CrashReporter { static func install(pendingURL: URL); static func collect(pendingURL: URL, outbox: ReportOutbox, engine: String, enabled: Bool) }`.
  - `install`:
    - `NSSetUncaughtExceptionHandler` writes `"\(name): \(reason)\n" + callStackSymbols.joined("\n")` to `pendingURL`.
    - Signal handlers for `SIGABRT, SIGSEGV, SIGBUS, SIGILL, SIGTRAP, SIGFPE` use only async-signal-safe calls: `open(2)` on a path C string prepared at install time, `write(2)` of a `"Signal N\n"` header built without allocation, `backtrace` + `backtrace_symbols_fd`, then restore `SIG_DFL` and `raise(sig)`.
    - Swift-runtime traps that bypass these are an accepted ceiling.
  - `collect`: if the pending file exists and isn't empty, and `enabled` is true, enqueue `Report.crash(signature: <first line>, details: <whole text>, engine:)`. When `enabled` is false, run `outbox.dropType("crash")`. Always delete the pending file.
- `pendingURL` = `ScanHistoryStore.dataDir.appendingPathComponent("crash-pending.txt")`. AppDelegate order at launch: `collect` → `install` → `Task { _ = await ReportOutbox.shared.flush(post: { await ReportOutbox.post($0, to: ReportOutbox.reportURL) }) }`.

- [ ] **Step 1: Failing tests**:
  - `testCollectTurnsPendingFileIntoOneCrash`: write `"Signal 11\n0 KhmerClipper \(NSHomeDirectory())/x"`. Expect one outbox file with `signature == "Signal 11"` and the home directory scrubbed, and the pending file gone.
  - `testCollectDisabledDropsCrashes`: pre-enqueue one crash and one feedback, write a pending file, then `enabled: false`. Only the feedback remains, and the pending file is gone.
- [ ] **Step 2: Run to verify fail** — `make -C apps/mac test`.
- [ ] **Step 3: Implement** the reporter, the prefs, the toggle, and the AppDelegate wiring.
- [ ] **Step 4: Run** `make -C apps/mac test` → PASS; `make -C apps/mac build` → succeeds.
- [ ] **Step 5: Commit** — `git commit -m "feat(mac): automatic crash reports and on-by-default capture images"`

---

### Task 8: Mac History layout A and Report sheet

**Files:**
- Create: `apps/mac/Sources/History/ReportView.swift`
- Modify: `apps/mac/Sources/History/HistoryView.swift`
- Modify: `apps/mac/Sources/History/ScanHistoryStore.swift`:
  - `flag(_ id: UUID, correction: String = "")` writes the merged feedback JSON
  - `captureURL(_ id: UUID) -> URL`
  - `captureMeta(_ id: UUID) -> [String: Any]`
- Create: `apps/mac/Sources/History/FeedbackMerge.swift` with `func mergeFeedback(_ data: Data, correction: String) -> Data`, which `ScanHistoryStore.flag` calls. Add it to `KhmerClipperTests.sources` in `apps/mac/project.yml` so the store itself stays out of the test target.

**Interfaces:**
- Consumes: `Report.capture`, `ReportOutbox.shared`, `ReportOutbox.post`, `ReportOutbox.reportURL`.
- Produces:
  - `struct ReportView: View { let imageURL: URL; let ocrText: String; let onSend: (String) -> Void; let onCancel: () -> Void }`. It shows the image (`NSImage(contentsOf:)`, max width 480), `KhmerClipper read` with the text, `What it should say` with a `TextEditor` prefilled with `ocrText`, the exact line `This image and text will be sent to the KhmerClipper developer.`, and Cancel / Send. `onSend` receives `""` when the trimmed edit equals the trimmed `ocrText`.
  - `HistoryView` rows are an `HStack` of:
    - a 72×40 thumbnail (when `store.hasCapture`)
    - the existing date/text column
    - `Button("Report")`, or a disabled `Button("Reported")` when `store.isFlagged`, shown only when a capture exists
    - Report opens `ReportView` in a `.sheet`. Remove the "Flag as Wrong" context-menu entry and the "Feedback Folder" header button. Keep the other context-menu items and Clear. The window frame widens to 520.
  - On send: `store.flag(id, correction:)`, then enqueue `Report.capture(pngURL: store.captureURL(id), meta: store.captureMeta(id), ocrText: (meta["ocr_text"] as? String) ?? record.text, correction:, engine: UserDefaults.standard.string(forKey: "ocrEngine") ?? "tesseract")`, then flush in a `Task`. Show an alert reading `Thanks — your report was sent.` or `Report saved. It will be sent when you're online.`

- [ ] **Step 1: Failing test** — `testMergeFeedbackAddsCorrection`: `{"ocr_text":"ក"}` with `"ខ"` decodes to `correction == "ខ"`, with `ocr_text` intact.
- [ ] **Step 2: Run to verify fail** — `make -C apps/mac test`.
- [ ] **Step 3: Implement** the view, the sheet, and the store helpers.
- [ ] **Step 4: Run** `make -C apps/mac test` → PASS; `make -C apps/mac build`. Then `make -C apps/mac run`, capture once, and open History: the thumbnail and Report button are visible, the Report sheet opens, and Cancel closes it.
- [ ] **Step 5: Commit** — `git commit -m "feat(mac): history thumbnails with Report button"`

---

### Task 9: Mac Send Feedback

**Files:**
- Create: `apps/mac/Sources/Reports/FeedbackView.swift`
- Modify: `apps/mac/Sources/App/AppDelegate.swift` (menu item `Send Feedback…` right after `History…`, opening a window that hosts `FeedbackView`, following the `showHistory` pattern)
- Modify: `apps/mac/project.yml` (add `FeedbackView.swift` to test sources)

**Interfaces:**
- Produces: `struct FeedbackView: View { let onSend: (String, String) -> Void; static func canSend(_ message: String) -> Bool }`. It has a `TextEditor` for the message and `TextField("Your email (if you'd like a reply)")`. Send is disabled unless `canSend`, and it sends trimmed values. AppDelegate enqueues `Report.feedback(...)`, flushes, shows the same two alerts, and closes the window.

- [ ] **Step 1: Failing test** — `testCanSendRequiresText`: `""` and `"  \n"` → false; `"សូមបន្ថែម"` → true.
- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Implement** the view and the menu item.
- [ ] **Step 4: Run** `make -C apps/mac test` → PASS; `make -C apps/mac build`.
- [ ] **Step 5: Commit** — `git commit -m "feat(mac): Send Feedback menu item"`

---

### Task 10: Deploy the relay and switch the apps on (with the developer)

This needs the developer's accounts. The executor prepares each command and pauses for every login or secret.

**Files:**
- Modify: `apps/report-relay/README.md` (the steps below, as a checklist)
- Modify: `apps/desktop/src/khmerclipper/reports.py` (`REPORT_URL`), `apps/mac/Sources/Reports/ReportOutbox.swift` (`reportURL`)

- [ ] **Step 1:** The developer creates the private repo: `gh repo create KimChengSHEANG/khmerclipper-reports --private --add-readme`, then `gh label create crash -R KimChengSHEANG/khmerclipper-reports` (and the same for `capture` and `feedback`).
- [ ] **Step 2:** The developer creates a fine-grained PAT limited to that repo (Issues RW, Contents RW) and a Resend API key on the account for `cheng900@gmail.com`.
- [ ] **Step 3:** In `apps/report-relay`, run `npx wrangler login`, `npx wrangler secret put GITHUB_TOKEN`, `npx wrangler secret put RESEND_API_KEY`, and `npx wrangler deploy`, then note the `*.workers.dev` URL.
- [ ] **Step 4: Smoke test** — `curl -s -X POST <url>/report -H 'X-KhmerClipper-Key: kc-reports-1' -H 'Content-Type: application/json' -d '{"type":"feedback","id":"'$(uuidgen | tr -d - | tr A-F a-f)'","app_version":"dev","os":"curl","engine":"none","message":"relay smoke test","email":""}'` → `{"ok":true,"via":"github"}`, and a `feedback` issue appears in the private repo.
- [ ] **Step 5:** Put the deployed URL into both constants. Run `make check-desktop` and `make -C apps/mac test`. The unconfigured-URL tests pass an explicit `REPLACE_ME` URL, so they keep passing.
- [ ] **Step 6: Commit** — `git commit -m "feat: point apps at the deployed report relay"`
