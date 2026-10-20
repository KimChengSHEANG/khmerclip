# KhmerClipper Rename Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the project, apps, identifiers, remote repos and local folder from KhmerClip to KhmerClipper as a clean break.

**Architecture:** Merge the reports branch into `main`, then do the rename on one branch in a separate worktree: path renames first (`git mv`), then a scripted case-preserving content replace over tracked text files, then review every remaining hit. Remote repos are renamed only after CI is green on the branch; the branch merges after that, then the local folder moves.

**Tech Stack:** git, Python 3 (desktop app, eval tools), Swift + XcodeGen (Mac app), PyInstaller/Inno Setup/dpkg packaging, GitHub CLI `gh`, `huggingface_hub`.

**Spec:** `docs/superpowers/specs/2026-10-20-khmerclipper-rename-design.md`

## Global Constraints

- Mapping (case preserved): `KhmerClip`→`KhmerClipper`, `khmerclip`→`khmerclipper`, `KHMERCLIP`→`KHMERCLIPPER`; bundle IDs `com.khmerclip.*`→`com.khmerclipper.*`; HF `kcsheang/khmerclip-trocr-khmer-onnx`→`kcsheang/khmerclipper-trocr-khmer-onnx`; GitHub `KimChengSHEANG/khmerclip`→`KimChengSHEANG/khmerclipper`, `KimChengSHEANG/khmerclip-reports`→`KimChengSHEANG/khmerclipper-reports`; relay `khmerclip-report-relay`→`khmerclipper-report-relay`.
- Clean break: no migration code, no `khmerclip` alias, no `KHMERCLIP_*` fallback, no `Replaces/Conflicts`.
- Windows installer gets a new `AppId` GUID.
- Do not rewrite git history or existing tags/releases. Never create a repo under an old name.
- The only "khmerclip" allowed in the tree afterwards: this plan and its spec (they describe the rename and are excluded from the replace), and the rename sentence in 0.3.0 release notes if written.
- No commit attribution lines.
- Kim's uncommitted change `training/eval/scripts/bench.py` in the main checkout must survive untouched.
- Other Claude sessions working in this repo are stopped before Task 2 starts (they would commit under old paths).

## Review Focus

- Binary files containing the byte string (PNG fixtures, ONNX/mlpackage, `.icns`): the replace must touch text files only (`git grep -I`), or it corrupts them.
- Running the replace twice must not produce `khmerclipperper`: the regex excludes an existing `per` suffix.
- Mixed spellings (`Khmer Clip`, `khmer-clip`, `khmer_clip`) are found by the case-insensitive `khmer[-_ ]?clip` scan in Task 3 Step 2 and fixed by hand.
- `git mv` of `khmerclip` into `khmerclipper` must be done per path, deepest first, not by a glob that re-matches the new name.
- Pinned HF model URLs keep their commit hash; after the HF rename the new URLs must serve files whose sha256 equals the manifest's `sha256`.

---

### Task 1: Merge the reports branch into main

**Files:** none edited by hand (merge commit).

**Interfaces:**
- Produces: `main` containing `feat/reports-history-feedback` (relay at `apps/report-relay/`, `reports.py`, `ReportOutbox.swift`).

- [ ] **Step 1:** In the reports worktree (`~/Sync/workspace/khmerclip-reports`), `git merge main` then `make check`. Expected: merge clean, all checks pass. If `make check` fails, stop and report to Kim (spec: "if that branch is not ready to merge, stop and come back").
- [ ] **Step 2:** In the main checkout, `git merge --no-ff feat/reports-history-feedback -m "merge: crash/feedback reports, history, feedback"`. Expected: clean merge; `bench.py` still modified.
- [ ] **Step 3:** `make check` in the main checkout. Expected: pass.
- [ ] **Step 4:** `git push origin main`.

### Task 2: Rename paths

**Files:** every tracked path containing `khmerclip` (case-insensitive), including:
- `apps/desktop/src/khmerclip/` → `apps/desktop/src/khmerclipper/`
- `training/eval/src/khmerclip_eval/` → `training/eval/src/khmerclipper_eval/`
- `apps/desktop/packaging/khmerclip.spec`, `packaging/linux/khmerclip.desktop`, `packaging/windows/khmerclip.iss`, icon files, docs file names (`*khmerclip*.md`)
- `apps/mac/KhmerClip.xcodeproj` (removed here, regenerated in Task 3)

**Interfaces:**
- Produces: branch `rename/khmerclipper` in worktree `~/Sync/workspace/khmerclip-rename`, all paths renamed, content untouched.

- [ ] **Step 1:** `git worktree add ~/Sync/workspace/khmerclip-rename -b rename/khmerclipper main`.
- [ ] **Step 2:** For each tracked path whose components contain the old name (deepest first), `git mv` it, changing only the component(s) with the old name, case-preserving. `git rm -r apps/mac/KhmerClip.xcodeproj`.
- [ ] **Step 3:** Verify: `git ls-files | grep -iE 'khmer[-_ ]?clip' | grep -viE 'khmer[-_ ]?clipper'` prints only the spec and this plan.
- [ ] **Step 4:** `git commit -m "refactor: rename paths to KhmerClipper"`. (Builds are broken until Task 3; expected on this branch.)

### Task 3: Rename content and identifiers

**Files:** all tracked text files with a hit, plus:
- Modify: `apps/desktop/packaging/windows/khmerclipper.iss` (`AppId` line)
- Regenerate: `apps/mac/KhmerClipper.xcodeproj` from `apps/mac/project.yml`

**Interfaces:**
- Consumes: Task 2 paths.
- Produces: package `khmerclipper` (import `khmerclipper`), CLI `khmerclipper`, eval package `khmerclipper_eval` / dist `khmerclipper-eval`, env vars `KHMERCLIPPER_*`, bundle IDs `com.khmerclipper.app|ocrbench|tests`, deb `khmerclipper` in `/opt/khmerclipper`, URLs at the new GitHub/HF names.

- [ ] **Step 1:** Replace over text files only, case-preserving, idempotent, skipping the spec and plan:

```bash
git grep -IlzE '[Kk][Hh][Mm][Ee][Rr][Cc][Ll][Ii][Pp]' -- . \
  ':!docs/superpowers/specs/2026-10-20-khmerclipper-rename-design.md' \
  ':!docs/superpowers/plans/2026-10-20-khmerclipper-rename.md' \
| xargs -0 perl -pi -e 's/KHMERCLIP(?!PER)/KHMERCLIPPER/g; s/KhmerClip(?!per)/KhmerClipper/g; s/khmerclip(?!per)/khmerclipper/g'
```

- [ ] **Step 2:** List leftovers: `git grep -niE 'khmer[-_ ]?clip' | grep -viE 'khmer[-_ ]?clipper'`. Fix each by hand. Expected after fixing: only the spec and this plan.
- [ ] **Step 3:** Replace the `AppId={{6E0B3C52-6C4E-4C1B-9C7B-6B2E7A4F1D10}` GUID with a fresh one from `uuidgen` (keep the `{{` Inno escaping).
- [ ] **Step 4:** `cd apps/mac && xcodegen generate`; `git add -A apps/mac/KhmerClipper.xcodeproj`. Verify `grep -c com.khmerclipper apps/mac/KhmerClipper.xcodeproj/project.pbxproj` ≥ 3.
- [ ] **Step 5:** Recreate local envs that embed the old name (delete `*.egg-info`, reinstall editable packages as the Makefile `check-*` targets expect).
- [ ] **Step 6:** `make check` in the rename worktree. Expected: all pass.
- [ ] **Step 7:** `make build-desktop` and `make build-mac`. Expected: artifacts named `KhmerClipper*`; `dist/KhmerClipper/` exists; `dist/KhmerClipper/KhmerClipper ocr ../mac/Tests/Fixtures/ocr/1.png` (from `apps/desktop`) prints `The quick brown fox`.
- [ ] **Step 8:** Binary check: `git diff --numstat main -- '*.png' '*.icns' '*.onnx'` shows `-	-` (binary, unchanged content) or nothing for every line.
- [ ] **Step 9:** `git commit -m "refactor: rename KhmerClip to KhmerClipper"`; push the branch; CI (`desktop`, `trivy`) green.

### Task 4: Rename remote repos

**Files:** none (remote operations); `.git/config` remote URL.

**Interfaces:**
- Consumes: Task 3 URLs already point at the new names.
- Produces: GitHub `KimChengSHEANG/khmerclipper`, `KimChengSHEANG/khmerclipper-reports`; HF `kcsheang/khmerclipper-trocr-khmer-onnx`.

- [ ] **Step 1:** Confirm with Kim right before running (outward-facing).
- [ ] **Step 2:** `gh repo rename khmerclipper -R KimChengSHEANG/khmerclip --yes`; `gh repo rename khmerclipper-reports -R KimChengSHEANG/khmerclip-reports --yes`.
- [ ] **Step 3:** `python -c "from huggingface_hub import move_repo; move_repo('kcsheang/khmerclip-trocr-khmer-onnx','kcsheang/khmerclipper-trocr-khmer-onnx')"` (needs Kim's HF token).
- [ ] **Step 4:** Verify: for each entry in `apps/desktop/src/khmerclipper/ocr/trocr_manifest.json`, download `url` and compare sha256 with the entry's `sha256`. Expected: all equal. `gh api repos/KimChengSHEANG/khmerclipper -q .name` prints `khmerclipper`.
- [ ] **Step 5:** `git remote set-url origin git@github.com:KimChengSHEANG/khmerclipper.git`.

### Task 5: Merge and move the local folder

- [ ] **Step 1:** In the main checkout: `git merge --ff-only rename/khmerclipper` (`--no-ff` if `main` moved), `make check`, `git push origin main`.
- [ ] **Step 2:** `git worktree remove ~/Sync/workspace/khmerclip-rename`; `git branch -d rename/khmerclipper`.
- [ ] **Step 3:** With no session running in them: `mv ~/Sync/workspace/khmerclip ~/Sync/workspace/khmerclipper`; `mv ~/Sync/workspace/khmerclip-reports ~/Sync/workspace/khmerclipper-reports`; `git -C ~/Sync/workspace/khmerclipper worktree repair ~/Sync/workspace/khmerclipper-reports`.
- [ ] **Step 4:** Verify: `git worktree list` shows no `prunable` entries; `git status` in the main checkout still shows `bench.py` modified.
- [ ] **Step 5:** Move `~/.claude/projects/-Users-kim-Sync-workspace-khmerclip` to `~/.claude/projects/-Users-kim-Sync-workspace-khmerclipper`.
- [ ] **Step 6:** Recreate virtualenvs in the moved folder (they embed absolute paths); `make check` passes there.

### Task 6: Replace Kim's own installs

- [ ] **Step 1:** Ubuntu 192.168.1.25: `sudo -n apt-get remove -y khmerclip`; install the `khmerclipper` deb from the Task 3 CI artifact; `/usr/bin/khmerclipper ocr` smoke test on the fixture image.
- [ ] **Step 2:** Give Kim the Mac steps: delete `/Applications/KhmerClip.app`, `~/Library/Application Support/KhmerClip`, the KhmerClip login item; install the new build; grant Screen Recording.
- [ ] **Step 3:** Remind Kim to remove `/etc/sudoers.d/claude-apt` on the Ubuntu machine.
