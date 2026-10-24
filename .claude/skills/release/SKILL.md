---
name: release
description: Cut and publish a KhmerClipper release - version bump, release notes, website update, Windows/Linux builds via CI, signed and notarized Mac DMG, GitHub release, optional Hugging Face model update. Use when asked to release, publish, ship or cut a new KhmerClipper version.
---

# Release KhmerClipper

One release ships four files on one GitHub release `v<ver>`:
- `KhmerClipper-<ver>.dmg` (Mac, built and notarized locally)
- `KhmerClipper-<ver>-windows-x64-setup.exe`, `khmerclipper_<ver>_amd64.deb`, `KhmerClipper-<ver>-x86_64.AppImage` (built by `.github/workflows/desktop.yml` when a `desktop-v<ver>` tag is pushed)

The GitHub Pages site (`site/`) redeploys on every push to `main` that touches `site/**`, and its Download button points at `releases/latest`.

Run the steps in order. Pushing to `main`, pushing the tag, uploading to Hugging Face and publishing the release are public: confirm each with the user before doing it, unless they already said to release.

## 1. Check the state

```bash
git switch main && git pull --ff-only && git status --short   # clean apart from untracked local tooling
(cd training/eval && PYTHONPATH=src .venv/bin/python -m pytest -q tests)
(cd apps/desktop && .venv/bin/python -m pytest -q)
(cd training/trocr && DYLD_LIBRARY_PATH=/opt/homebrew/lib PYTHONPATH=src .venv/bin/python -m pytest -q tests)
(cd apps/mac && make test)
```

All four must pass. `DYLD_LIBRARY_PATH` lets Pillow find Homebrew's fribidi; without it the render tests skip.

## 2. Pick the version

```bash
git describe --tags --abbrev=0 --match 'desktop-v*'
git log --oneline <that tag>..HEAD
```

New user-visible features → minor bump (0.3.0 → 0.4.0). Fixes only → patch (0.3.0 → 0.3.1). Show the user the commit list and the proposed version, and confirm.

## 3. Bump the version

The version lives in exactly three places:
- `apps/desktop/src/khmerclipper/__init__.py`: `__version__ = "<ver>"`. The CI release job fails if the tag isn't `desktop-v<__version__>`.
- `apps/mac/Resources/Info.plist`: the `CFBundleShortVersionString` string.
- `apps/mac/project.yml`: `CFBundleShortVersionString: "<ver>"`.

## 4. Check the models

- **Mac (CoreML).** If `apps/mac/tools/trocr/*.py` or the checkpoint changed since the last release, run `cd apps/mac && make trocr-khmer`. The model's inputs and outputs must match `TrOCREngine.swift`. Today the encoder emits `cross_k`/`cross_v`; check with:
  ```bash
  strings "$(find apps/mac/Vendor/trocr/khmer/encoder.mlpackage -name '*.mlmodel')" | grep -oE 'cross_k|cross_v|encoder_hidden_states' | sort -u
  ```
  A mismatch fails every TrOCR capture. The compiled-model cache name in `TrOCREngine.swift` (`\(size.rawValue)-xkv`) must change whenever the model I/O changes, or users keep a stale cache.
- **Windows/Linux (ONNX).** If `apps/desktop/src/khmerclipper/ocr/trocr_manifest.json` changed since the last tag, every desktop user downloads the model again (about 1.5 GB). The release notes must say so (step 5).

## 5. Write the release notes

Copy the previous `docs/releases/v<prev>.md` to `docs/releases/v<ver>.md` and keep its sections: intro line, **What's new**, **Upgrading** (when anything changes for existing users), **Download** table, **Install**, **Use**. Replace every version string.

- Write for people who use the app, not developers: what they'll notice, in plain words.
- Use only numbers measured in `training/eval/results/*.md` or a fresh benchmark, and state them no stronger than measured. "Up to twice as fast" for a 2× best case, not "half the time".
- CI reads this file as the GitHub release body, so commit it before pushing the tag.

## 6. Update the website

Edit `site/index.html` in the same release commit:
- Replace the **What's new in <prev>** section (`id="new"`) with **What's new in <ver>**: the release date, 2-4 cards matching the notes, and a link to `releases/tag/v<ver>`.
- Update `Version <ver>` in the hero's platform line.
- Refresh any stat or chart that changed (the `#accuracy` stats, How it works), from the same results files.
- Preview before committing. The iframe trick gets a true 390 px phone width out of headless Chrome, whose window can't be that narrow:
  ```bash
  C="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
  "$C" --headless=new --disable-gpu --hide-scrollbars --window-size=1280,1500 --screenshot=/tmp/site-desk.png "file://$PWD/site/index.html"
  printf '<iframe src="file://%s/site/index.html" width="390" height="1900" style="border:0"></iframe>' "$PWD" > /tmp/w.html
  "$C" --headless=new --disable-gpu --hide-scrollbars --allow-file-access-from-files --window-size=420,1900 --screenshot=/tmp/site-phone.png file:///tmp/w.html
  ```
  Look at both screenshots: no horizontal overflow, no orphaned cards.

## 7. Commit, push, tag

```bash
git add apps/desktop/src/khmerclipper/__init__.py apps/mac/Resources/Info.plist apps/mac/project.yml \
        docs/releases/v<ver>.md site/index.html
git commit -m "release: <ver>"
git push origin main
git tag desktop-v<ver> && git push origin desktop-v<ver>
```

The tag push starts the `desktop` workflow. It tests and builds on Ubuntu and Windows, then creates a **draft** release `v<ver>` with three assets. The `main` push redeploys the site.

## 8. Build the Mac DMG

Run this while CI builds:

```bash
PROFILE=KhmerClipper
xcrun notarytool history --keychain-profile "$PROFILE" >/dev/null 2>&1 || PROFILE=khmerclip
SIGN_ID="Developer ID Application: Ratanak JAI (VJ6RKAA654)" NOTARY_PROFILE=$PROFILE make release-mac
```

It takes about 10-20 minutes, mostly notarization; run it in the background. Then verify all three:

```bash
D=apps/mac/dist/KhmerClipper-<ver>.dmg
spctl -a -t open --context context:primary-signature -v "$D"   # must say: source=Notarized Developer ID
xcrun stapler validate "$D"                                      # The validate action worked!
/usr/libexec/PlistBuddy -c 'Print CFBundleShortVersionString' \
  apps/mac/build/Build/Products/Release/KhmerClipper.app/Contents/Info.plist   # <ver>
```

`VJ6RKAA654` is the team that signed every earlier release. Never sign with a different identity: users' Screen Recording permission is tied to the signature.

## 9. Smoke-test the Mac build (extra A)

Before uploading, run OCR with the release build's code on one known image and compare with its label:

```bash
cd apps/mac && make ocrbench
build/Build/Products/Release/OCRBench --engine trocr ../../training/eval/fixtures/ui/ui-033.png   # expect រកមិនឃើញទិន្នន័យ
```

If it prints nothing, garbage or an error about `cross_k`, stop: the models and code don't match (step 4).

## 10. Publish

```bash
gh run watch <desktop run id for the tag> --exit-status
gh release view v<ver> --json isDraft,assets --jq '.isDraft, (.assets[] | .name)'   # draft, 3 assets
gh release upload v<ver> apps/mac/dist/KhmerClipper-<ver>.dmg
gh release edit v<ver> --draft=false --latest
gh release view v<ver> --json isDraft,assets --jq '.isDraft, (.assets[] | "\(.name) \(.size)")'   # false, 4 assets
curl -sIL https://github.com/KimChengSHEANG/khmerclipper/releases/latest/download/KhmerClipper-<ver>.dmg | grep -E '^HTTP'   # 200
```

Then confirm the site:

```bash
gh run list -w pages -L 1
curl -s https://kimchengsheang.github.io/khmerclipper/ | grep -c "What's new in <ver>"   # 1
```

## 11. Hugging Face (extra B, only when the weights changed)

If this release ships new model weights (a new checkpoint, not just an export change):
- Upload the PyTorch weights to `kcsheang/khmerclipper-trocr-khmer`, from `apps/mac/raw_models/TrOCR-khmer`, and update the model card's results and "This release is vN" line.
- The ONNX export goes up with `apps/desktop/tools/publish_models.py`. It pins a new commit in `trocr_manifest.json`, which must then be committed before step 7.

Load the downloaded model once with `transformers` and read one fixture to confirm the upload.

## 12. README changelog (extra C)

Add one line for the release to the README's results section when accuracy or speed numbers changed, linking `docs/releases/v<ver>.md`. Keep the README's numbers in step with the website's.

## Gotchas

- **Notary profile name.** It was created as `khmerclip` before the rename. Step 8 falls back to it until `KhmerClipper` exists. Create the new one once with `xcrun notarytool store-credentials KhmerClipper --apple-id <id> --team-id VJ6RKAA654`; it prompts for the app-specific password. `security` can't copy the old item, because notarytool keeps it in the protected keychain.
- **Draft release.** Never publish while the draft has fewer than 4 files; users would get a release without their platform's download.
- **Tag and version.** They must match exactly (`desktop-v0.3.0` ↔ `__version__ = "0.3.0"`), or the CI release job fails after the builds. To redo: delete the tag locally and on origin, fix, re-tag.
- **zsh.** An unquoted `$var` holding several words is one argument in zsh. Use `${=var}` or arrays in loops.
- **macOS rsync** is openrsync: no `--info=progress2`.
- **Leftover drafts** from re-runs pile up on the releases page (`gh release list`). Offer to delete duplicates; never delete a published release.
- **CI failures that aren't yours.** Trivy fails when a new CVE lands in a locked dependency, and an Ubuntu job exit code 100 is an `apt` network error. Read the log before retrying; fix the dependency in its own PR.
