# Rename KhmerClip → KhmerClipper

Date: 2026-10-20

## Intent

The project, every app, and every internal identifier is renamed from KhmerClip to KhmerClipper.

0.1/0.2 are released but nobody uses them yet, so this is a **clean break**: no migration code, no
compatibility aliases, no old names kept alive for old installs. After the rename, "khmerclip" exists only
in git history and in the already-published 0.1.x/0.2.0 tags and releases.

## Goals

- Every name changes: display name, Python packages, the CLI command, file and directory names, packaging,
  bundle IDs, data directories, CI, docs (historical specs, plans and eval logs included), GitHub and
  Hugging Face repos, the local checkout folder.
- The crash/feedback reports feature (branch `feat/reports-history-feedback`) ends up under the new names
  before its relay is ever deployed.

## Non-goals

- No migration of user data, settings, permissions or launch-at-login entries from KhmerClip.
- No `khmerclip` command alias, no `KHMERCLIP_*` env-var fallback, no `Replaces/Conflicts` package metadata.
- No functional changes beyond the rename (the reports feature lands as already built on its branch).
- No new icon or visual rebrand.
- Git history and the existing `v0.1.x`/`v0.2.0`/`desktop-v0.2.0` tags and releases are not rewritten.

## Anti-goals

- Any "khmerclip" left in the working tree after the rename (case-insensitive, any separator), other than
  intentional mentions in the release notes explaining the rename.
- A broken build, test suite, or packaging job on any platform.
- Creating a repo under an old name later (it would hijack GitHub's redirect from the old URL).

## Decisions (made by Kim)

- Rename everything, internal names included; historical docs rewritten too.
- Nobody uses 0.1/0.2 yet: Claude decides the approach. The calls below follow from that.

## Name mapping

| Old | New |
|---|---|
| KhmerClip (display, product, `.app`, `.exe`, AppImage, dmg, data dirs) | KhmerClipper |
| `khmerclip` (Python package, CLI, deb package, `/opt/khmerclip`, file names) | `khmerclipper` |
| `khmerclip_eval` / `khmerclip-eval` | `khmerclipper_eval` / `khmerclipper-eval` |
| `com.khmerclip.app` / `.ocrbench` / `.tests` | `com.khmerclipper.app` / `.ocrbench` / `.tests` |
| `apps/mac/KhmerClip.xcodeproj`, `project.yml` `name: KhmerClip` | `KhmerClipper.xcodeproj`, `name: KhmerClipper` |
| `KHMERCLIP_*` env vars | `KHMERCLIPPER_*` |
| `khmerclip-report-relay` (Cloudflare Worker, not yet deployed) | `khmerclipper-report-relay` |
| GitHub `KimChengSHEANG/khmerclip` | `KimChengSHEANG/khmerclipper` |
| GitHub `KimChengSHEANG/khmerclip-reports` (private) | `KimChengSHEANG/khmerclipper-reports` |
| HF `kcsheang/khmerclip-trocr-khmer-onnx` | `kcsheang/khmerclipper-trocr-khmer-onnx` |
| `~/Sync/workspace/khmerclip` (and sibling worktree dirs such as `khmerclip-reports`) | `~/Sync/workspace/khmerclipper` (`khmerclipper-reports`, …) |

Case is preserved: `KhmerClip`→`KhmerClipper`, `khmerclip`→`khmerclipper`, `KHMERCLIP`→`KHMERCLIPPER`.
The Windows installer gets a **new** `AppId` GUID (fresh product, nothing to upgrade from).

## Order

1. **Merge `feat/reports-history-feedback` into `main` first**, so the rename covers it in one pass and the
   relay is named `khmerclipper-report-relay` from its first deploy. If that branch is not ready to merge,
   stop and come back (the alternative is renaming `main` and rebasing that branch afterwards).
2. **Rename in the repo** on a branch: files and directories (`git mv`), then content, then review every
   remaining hit. Regenerate the Xcode project from `project.yml` rather than hand-editing `project.pbxproj`
   where the project uses XcodeGen; recreate venvs and `*.egg-info` rather than editing them.
3. **Tests and builds pass** on all platforms (CI: desktop matrix, Mac build/tests), including packaging
   jobs producing `KhmerClipper-*` artifacts and the `khmerclipper` deb.
4. **Rename remote repos** (GitHub `khmerclip` → `khmerclipper`, `khmerclip-reports` →
   `khmerclipper-reports`, HF model repo) and update in-repo URLs to the new names directly: updater feed,
   release links, HF manifest URLs (same pinned commit hash; verify they download with the same sha256),
   `publish_models.py`, deb `Homepage`, relay `REPORTS_REPO`. Update the local git remote.
5. **Merge to `main`**, then move the local folder (below).
6. **Release 0.3.0** (`desktop-v0.3.0`, "KhmerClipper 0.3.0") when Kim wants a release; notes say the app
   was renamed. Existing 0.1.x/0.2.0 releases stay as they are.

## Local folder, checkouts, and Kim's own installs

- Move `~/Sync/workspace/khmerclip` → `~/Sync/workspace/khmerclipper` (and sibling worktree directories);
  run `git worktree repair`; recreate virtualenvs.
- Other feature branches (`feat/*`, `fix/*`, `worktree-*`) are not renamed now; whoever picks one up next
  rebases it onto the renamed `main`.
- Claude Code's per-project memory directory is keyed by the folder path; move it to the new key.
- Remove the 0.2.0 KhmerClip installs Kim has (Ubuntu at 192.168.1.25: `apt remove khmerclip`; Mac:
  delete `KhmerClip.app`, its `~/Library/Application Support/KhmerClip`, and its login item) and install
  KhmerClipper fresh once built.

## Left to the builder

- Exact tooling for the replace (scripted, with every hit reviewed), commit granularity, test strategy.
- Handling of generated or vendored files (`apps/mac/Vendor`, `raw_models`, lock files): rename only what
  is project-owned.

## Calls made by Claude (please check)

- Clean break, no migration or compatibility shims, because no one uses 0.1/0.2.
- Merge the reports branch before renaming.
- New Windows `AppId` GUID.
- Version 0.3.0 for the first KhmerClipper release.
- Old tags/releases and git history left untouched; other branches rebased later by whoever needs them.
