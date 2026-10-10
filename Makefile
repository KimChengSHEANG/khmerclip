# `make check` runs every test suite in the repo. Each Python project runs in
# its own throwaway uv env built from its pyproject, so a stale editable
# install elsewhere (e.g. from a worktree) or the active pyenv/venv can't leak
# packages in. Python 3.12 is what apps/desktop requires (<3.13).
UVT = uv run --isolated --no-project --managed-python --python 3.12 --with pytest --with-editable

.PHONY: build build-mac build-desktop check check-eval check-tesseract check-trocr check-desktop check-mac-tools check-mac

# `make build` builds every app that can be built on this machine.
build: build-mac

# Debug KhmerClip.app -> apps/mac/build/Build/Products/Debug/. Release/notarized
# archive steps are in apps/mac/README.md.
build-mac:
	$(MAKE) -C apps/mac build

# ponytail: stub until the desktop GUI (plan Tasks 5-15) and packaging (Task 16) land.
# PyInstaller can't cross-compile, so Windows/Linux artifacts will come from CI (Task 17).
build-desktop:
	@echo "apps/desktop has no GUI or packaging yet -- see docs/PROJECT.md §10" >&2; exit 1

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
