# `make check` runs every test suite in the repo. Each Python project runs in
# its own throwaway uv env built from its pyproject, so a stale editable
# install elsewhere (e.g. from a worktree) or the active pyenv/venv can't leak
# packages in. Python 3.12 is what apps/desktop requires (<3.13).
UVT = uv run --isolated --no-project --managed-python --python 3.12 --with pytest --with-editable

.PHONY: check check-eval check-tesseract check-trocr check-desktop check-mac-tools check-mac

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
