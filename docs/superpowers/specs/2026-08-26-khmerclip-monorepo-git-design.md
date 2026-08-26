# KhmerClip monorepo git organization

**Date:** 2026-08-26  
**Status:** approved design (pending user review of this file)  
**Goal:** One private GitHub repo holding the macOS app + EasyOCR/TrOCR training projects, with full history preserved under clean paths.

## Decisions

| Topic | Choice |
|-------|--------|
| History | Preserve all three histories (`git filter-repo` + unrelated-history merges) |
| Layout | `apps/mac/`, `training/easyocr/`, `training/trocr/` |
| Dirty trees | Commit in each source repo before rewrite |
| Remote | New **private** GitHub repo (prefer name `khmerclip`) |
| Approach | filter-repo rewrite into subdirectory prefixes, then merge |

## Target layout

```
khmerclip/                    # git root, branch main
  README.md                   # product overview + map to subprojects
  .gitignore                  # union of the three + root junk
  docs/superpowers/specs/     # this design (and later plans)
  apps/mac/                   # was mac/
  training/easyocr/           # was EasyOCR-KhmerClip/
  training/trocr/             # was TrOCR-KhmerClip/
```

### Ignore / never commit

Root `.gitignore` is a safety net; keep existing per-project gitignores under their new prefixes.

Must stay out of git:

- `apps/mac/build/` (~2.8G)
- `apps/mac/raw_models/` (~1.2G)
- `apps/mac/Vendor/` weights/venvs (~2.3G), except the small EasyOCR scripts already allowlisted in that project's `.gitignore`
- Python venvs, `__pycache__/`, experiment outputs
- `.env*`
- `.DS_Store`, `.claude/`, `.cursor/`, `.devswarm-temp/`

### Nested repos

After the monorepo is the active checkout, there is a single `.git` at `khmerclip/`. Nested `.git` directories from the old three projects are not kept in the final tree.

## Merge procedure

Work in a **scratch directory** (e.g. `/tmp/khmerclip-merge`) until verified, then replace the workspace.

### Prerequisites

- `git-filter-repo` installed (`brew install git-filter-repo` or `pip install git-filter-repo`) — **not currently on PATH**
- `gh` authenticated for private repo create (`gh` is available)
- Local backup of the three project folders until first successful push + smoke check

### Steps

1. **Commit dirty work** in `mac/` and `TrOCR-KhmerClip/` so uncommitted edits are in history before rewrite.
2. **Copy/clone** each repo into scratch; run `git filter-repo --to-subdirectory-filter <prefix>`:
   - mac → `apps/mac/`
   - EasyOCR → `training/easyocr/`
   - TrOCR → `training/trocr/`
3. **Init** monorepo on `main`; first commit = root `README.md` + root `.gitignore` (+ this design doc if not already present).
4. **Merge** each rewritten history (`git remote add` + `git merge --allow-unrelated-histories`).
5. **Sanity checks:**
   - `git log --follow` on at least one file under each prefix
   - large artifact dirs still ignored / not staged
   - no nested `.git` in the working tree
6. **Swap** verified tree into `khmerclip/` (keep pre-merge folders as backup until confirmed).
7. **GitHub:** create private repo and push, e.g. `gh repo create khmerclip --private --source=. --remote=origin` then `git push -u origin main` (or create empty remote then push). If name taken, pick a close variant once, then ask.

### Risk notes

- filter-repo **rewrites SHAs** (expected). None of the three projects have remotes today → low external collision risk.
- Do not force-push or delete anything on GitHub beyond normal private create/push for this new repo.

## After merge

### Root README

Short overview:

- KhmerClip = macOS menu-bar OCR app (Khmer + mixed Khmer/English)
- `apps/mac` = app
- `training/easyocr` and `training/trocr` = training pipelines that feed app engines
- Link to each subfolder README for build/train details
- Note that large models/build artifacts are not in git; use each project’s `make` / scripts

### Path updates

Minimal only: fix Makefiles / READMEs / scripts that hardcode old folder names (`mac/`, `EasyOCR-KhmerClip/`, `TrOCR-KhmerClip/`) when those refs break after the move. No drive-by refactors.

### Cleanup

- Single git root; nested `.git` gone from the active tree
- Keep local backup of pre-merge folders until push + smoke check (`make` / open Xcode under `apps/mac`)
- Never commit secrets, weights, or `build/`

## Out of scope

- CI
- Monorepo build tooling (nx, bazel, etc.)
- Renaming Python packages / import paths inside training code
- Publishing or uploading model weights

## Success criteria

- One private GitHub remote on `main`
- All three histories reachable under the new prefixes
- Working tree size dominated by source, not `build/` / `Vendor/` / `raw_models/`
- Subprojects still build/train via their existing docs after any minimal path fixes
)