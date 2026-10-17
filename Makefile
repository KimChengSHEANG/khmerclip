# `make check` runs every test suite in the repo. Each Python project runs in
# its own throwaway uv env built from its pyproject, so a stale editable
# install elsewhere (e.g. from a worktree) or the active pyenv/venv can't leak
# packages in. Python 3.12 is what apps/desktop requires (<3.13).
UVT = uv run --isolated --no-project --managed-python --python 3.12 --with pytest --with-editable

.PHONY: run build build-mac build-desktop release-mac check check-eval check-tesseract check-trocr check-desktop check-mac-tools check-mac

# `make run` builds and launches the Mac app.
run:
	$(MAKE) -C apps/mac run

# `make build` builds every app that can be built on this machine.
build: build-mac

# Debug KhmerClip.app -> apps/mac/build/Build/Products/Debug/. Release/notarized
# archive steps are in apps/mac/README.md.
build-mac:
	$(MAKE) -C apps/mac build

# Signed + notarized apps/mac/dist/KhmerClip-<ver>.dmg (Tesseract + TrOCR). Needs SIGN_ID
# ("Developer ID Application: ...") and NOTARY_PROFILE; without them it's ad-hoc signed.
# See apps/mac/tools/release.sh.
release-mac:
	apps/mac/tools/release.sh

# Frozen app for THIS OS -> apps/desktop/dist/KhmerClip/. PyInstaller can't cross-compile:
# the Windows installer and Linux AppImage come from .github/workflows/desktop.yml
# (run it manually, or push a desktop-v<version> tag to publish a release).
build-desktop:
	cd apps/desktop && $(UVT) '.[dev]' python tools/fetch_tessdata.py
	cd apps/desktop && $(UVT) '.[dev]' pyinstaller packaging/khmerclip.spec --noconfirm

check: check-eval check-tesseract check-trocr check-desktop check-mac-tools check-mac
	@echo "make check: all suites passed"

check-eval:
	cd training/eval && $(UVT) . pytest -q

check-tesseract:
	cd training/tesseract && $(UVT) . pytest -q

check-trocr:
	cd training/trocr && $(UVT) . --with torch pytest -q

check-desktop:
	cd apps/desktop && $(UVT) '.[dev]' pytest -q

# CoreML converter tests (tools/trocr) and the Swift app tests.
check-mac-tools:
	$(MAKE) -C apps/mac trocr-test

check-mac:
	$(MAKE) -C apps/mac test
