# KhmerClipper Monorepo Git Organization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge `mac/`, `EasyOCR-KhmerClipper/`, and `TrOCR-KhmerClipper/` into one private GitHub monorepo at `khmerclipper/` with histories preserved under `apps/mac/`, `training/easyocr/`, and `training/trocr/`.

**Architecture:** Commit dirty work in each source repo, rewrite each history with `git filter-repo --to-subdirectory-filter`, init a fresh monorepo with root README/gitignore, merge the three rewritten histories with `--allow-unrelated-histories`, verify ignores and history, then create a private GitHub remote and push `main`.

**Tech Stack:** git, git-filter-repo, GitHub CLI (`gh`), existing Xcode/Python project trees (unchanged internals)

**Spec:** `docs/superpowers/specs/2026-08-26-khmerclipper-monorepo-git-design.md`

## Global Constraints

- Preserve all three git histories (no squash-only / discard history)
- Target prefixes: `apps/mac/`, `training/easyocr/`, `training/trocr/`
- Commit dirty trees in source repos before rewrite
- New GitHub remote must be **private**; preferred name `khmerclipper`
- Never commit `build/`, `raw_models/`, `Vendor/` weights/venvs, `.env*`, or other large artifacts listed in the spec
- Work in scratch dir `/tmp/khmerclipper-merge` until verified; keep local backup of pre-merge folders until push + smoke check
- Minimal path-reference fixes only; no CI, no package renames, no model publishing

## File structure (create / modify)

| Path | Responsibility |
|------|----------------|
| `README.md` (root) | Product overview + map to three subprojects |
| `.gitignore` (root) | Safety-net ignores for OS/IDE/secrets + mac artifact paths |
| `docs/superpowers/specs/2026-08-26-khmerclipper-monorepo-git-design.md` | Carry into monorepo first commit |
| `docs/superpowers/plans/2026-08-26-khmerclipper-monorepo-git.md` | This plan (carry into monorepo) |
| `apps/mac/**` | Rewritten history of current `mac/` |
| `training/easyocr/**` | Rewritten history of current `EasyOCR-KhmerClipper/` |
| `training/trocr/**` | Rewritten history of current `TrOCR-KhmerClipper/` |
| Path refs inside subproject Makefiles/READMEs | Fix only if they break after move |

---

### Task 1: Install prerequisites

**Files:**
- None (tooling only)

**Interfaces:**
- Consumes: none
- Produces: `git-filter-repo` and authenticated `gh` available on PATH

- [ ] **Step 1: Install git-filter-repo**

```bash
brew install git-filter-repo
```

Expected: command succeeds (or brew reports already installed).

- [ ] **Step 2: Verify tools**

```bash
which git-filter-repo
git-filter-repo --version
which gh
gh auth status
```

Expected: `git-filter-repo` on PATH; `gh` logged in to GitHub with rights to create a private repo.

- [ ] **Step 3: Stop if `gh` not authenticated**

If `gh auth status` fails, run `gh auth login` interactively and re-check before continuing. Do not proceed to Task 7 without auth.

---

### Task 2: Commit dirty work in source repos

**Files:**
- Modify (stage/commit): everything dirty under `mac/` and `TrOCR-KhmerClipper/` (see statuses below)
- `EasyOCR-KhmerClipper/`: expect clean; verify only

**Interfaces:**
- Consumes: current working trees at `/Users/kim/Sync/workspace/khmerclipper/{mac,TrOCR-KhmerClipper,EasyOCR-KhmerClipper}`
- Produces: clean `git status` on all three before any filter-repo copy

- [ ] **Step 1: Verify EasyOCR is clean**

```bash
cd /Users/kim/Sync/workspace/khmerclipper/EasyOCR-KhmerClipper
git status --porcelain
```

Expected: empty output. If not empty, stage and commit with an accurate message before continuing.

- [ ] **Step 2: Commit TrOCR dirty files**

```bash
cd /Users/kim/Sync/workspace/khmerclipper/TrOCR-KhmerClipper
git status
git add scripts/evaluate_model.py scripts/train.py src/trocr_khmer/__init__.py src/trocr_khmer/dataset.py src/trocr_khmer/model.py
git commit -m "$(cat <<'EOF'
Save WIP TrOCR training/eval and dataset updates before monorepo merge.

EOF
)"
git status --porcelain
```

Expected: empty porcelain after commit. Adjust message if the diff is clearly something else (read `git diff --stat` first).

- [ ] **Step 3: Commit mac dirty files (KhmerClipper rename + related)**

```bash
cd /Users/kim/Sync/workspace/khmerclipper/mac
git status
git add -A
git status
```

Confirm staged set does **not** include `build/`, `raw_models/`, or bulk `Vendor/` weights. If those appear, unstage and fix `.gitignore` before committing.

```bash
git commit -m "$(cat <<'EOF'
Save KhmerClipper rename and OCR updates before monorepo merge.

EOF
)"
git status --porcelain
```

Expected: empty porcelain (aside from ignored build artifacts).

- [ ] **Step 4: Record tip SHAs for later sanity**

```bash
cd /Users/kim/Sync/workspace/khmerclipper/mac && git rev-parse HEAD
cd /Users/kim/Sync/workspace/khmerclipper/EasyOCR-KhmerClipper && git rev-parse HEAD
cd /Users/kim/Sync/workspace/khmerclipper/TrOCR-KhmerClipper && git rev-parse HEAD
```

Save the three SHAs in the agent transcript / notes (pre-rewrite tips).

---

### Task 3: Scratch copies + filter-repo path rewrite

**Files:**
- Create (scratch only): `/tmp/khmerclipper-merge/{mac,easyocr,trocr}/` rewritten clones

**Interfaces:**
- Consumes: clean source repos from Task 2
- Produces: three rewritten git dirs whose trees are rooted at `apps/mac/`, `training/easyocr/`, `training/trocr/`

- [ ] **Step 1: Create scratch workspace**

```bash
rm -rf /tmp/khmerclipper-merge
mkdir -p /tmp/khmerclipper-merge
```

- [ ] **Step 2: Copy repos into scratch (full `.git`)**

```bash
cp -a /Users/kim/Sync/workspace/khmerclipper/mac /tmp/khmerclipper-merge/mac
cp -a /Users/kim/Sync/workspace/khmerclipper/EasyOCR-KhmerClipper /tmp/khmerclipper-merge/easyocr
cp -a /Users/kim/Sync/workspace/khmerclipper/TrOCR-KhmerClipper /tmp/khmerclipper-merge/trocr
```

- [ ] **Step 3: Rewrite mac → `apps/mac/`**

```bash
cd /tmp/khmerclipper-merge/mac
git filter-repo --to-subdirectory-filter apps/mac/ --force
git log -1 --name-only
ls apps/mac | head
```

Expected: latest commit lists paths under `apps/mac/`; `apps/mac/Sources` (or equivalent) exists. No files at the old repo root except `.git`.

- [ ] **Step 4: Rewrite EasyOCR → `training/easyocr/`**

```bash
cd /tmp/khmerclipper-merge/easyocr
git filter-repo --to-subdirectory-filter training/easyocr/ --force
git log -1 --name-only
ls training/easyocr | head
```

Expected: paths under `training/easyocr/`; `training/easyocr/pyproject.toml` exists.

- [ ] **Step 5: Rewrite TrOCR → `training/trocr/`**

```bash
cd /tmp/khmerclipper-merge/trocr
git filter-repo --to-subdirectory-filter training/trocr/ --force
git log -1 --name-only
ls training/trocr | head
```

Expected: paths under `training/trocr/`; `training/trocr/pyproject.toml` exists.

---

### Task 4: Init monorepo with root README + .gitignore

**Files:**
- Create: `/tmp/khmerclipper-merge/mono/README.md`
- Create: `/tmp/khmerclipper-merge/mono/.gitignore`
- Copy into mono: design + plan under `docs/superpowers/`

**Interfaces:**
- Consumes: design/plan from workspace `docs/superpowers/`
- Produces: monorepo at `/tmp/khmerclipper-merge/mono` on `main` with initial commit

- [ ] **Step 1: Init repo**

```bash
mkdir -p /tmp/khmerclipper-merge/mono
cd /tmp/khmerclipper-merge/mono
git init -b main
```

- [ ] **Step 2: Write root `.gitignore`**

Create `/tmp/khmerclipper-merge/mono/.gitignore` with exactly:

```gitignore
# OS / editors
.DS_Store
Thumbs.db
.idea/
.vscode/
*.swp
*.swo
*~

# Agent / local tooling
.claude/
.cursor/
.devswarm-temp/

# Secrets
.env
.env.*
!.env.example

# macOS app build + heavy artifacts (also covered under apps/mac/.gitignore)
**/DerivedData/
*.xcuserstate
apps/mac/build/
apps/mac/raw_models/
apps/mac/Vendor/
!apps/mac/Vendor/easyocr/install_khmer_model.py
!apps/mac/Vendor/easyocr/recognize.py
!apps/mac/Vendor/easyocr/requirements.txt
!apps/mac/Vendor/easyocr/khmer/
!apps/mac/Vendor/easyocr/khmer/**

# Python (safety net; training projects keep their own gitignores)
__pycache__/
*.py[cod]
.venv/
venv/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.cache/
.huggingface/
wandb/
*.egg-info/
dist/
```

- [ ] **Step 3: Write root `README.md`**

Create `/tmp/khmerclipper-merge/mono/README.md` with exactly:

```markdown
# KhmerClipper

macOS menu-bar OCR for Khmer (and mixed Khmer/English) text — screen capture, file OCR, and on-device engines.

## Layout

| Path | What it is |
|------|------------|
| [`apps/mac`](apps/mac) | KhmerClipper macOS app (Swift / Xcode) |
| [`training/easyocr`](training/easyocr) | EasyOCR-compatible Khmer recognition training |
| [`training/trocr`](training/trocr) | TrOCR Khmer fine-tuning |

Training outputs feed the app’s EasyOCR and TrOCR engines. Large build products and model weights are **not** in git — follow each subproject README / `Makefile` to generate them locally.

## Quick start (app)

See [`apps/mac/README.md`](apps/mac/README.md): install `xcodegen`, then `make run` from `apps/mac`.

## Training

- EasyOCR: [`training/easyocr/README.md`](training/easyocr/README.md)
- TrOCR: [`training/trocr/README.md`](training/trocr/README.md)
```

- [ ] **Step 4: Copy design + plan docs**

```bash
mkdir -p /tmp/khmerclipper-merge/mono/docs/superpowers/specs
mkdir -p /tmp/khmerclipper-merge/mono/docs/superpowers/plans
cp /Users/kim/Sync/workspace/khmerclipper/docs/superpowers/specs/2026-08-26-khmerclipper-monorepo-git-design.md \
  /tmp/khmerclipper-merge/mono/docs/superpowers/specs/
cp /Users/kim/Sync/workspace/khmerclipper/docs/superpowers/plans/2026-08-26-khmerclipper-monorepo-git.md \
  /tmp/khmerclipper-merge/mono/docs/superpowers/plans/
```

- [ ] **Step 5: Initial commit**

```bash
cd /tmp/khmerclipper-merge/mono
git add README.md .gitignore docs
git commit -m "$(cat <<'EOF'
Initial monorepo scaffold with README, gitignore, and design docs.

EOF
)"
git log -1 --oneline
```

Expected: one commit on `main` containing only root scaffolding files.

---

### Task 5: Merge rewritten histories into monorepo

**Files:**
- Modify: `/tmp/khmerclipper-merge/mono/.git` history (merges)

**Interfaces:**
- Consumes: rewritten remotes from Task 3; scaffold from Task 4
- Produces: single `main` containing all three project trees + histories

- [ ] **Step 1: Add remotes and fetch**

```bash
cd /tmp/khmerclipper-merge/mono
git remote add mac /tmp/khmerclipper-merge/mac
git remote add easyocr /tmp/khmerclipper-merge/easyocr
git remote add trocr /tmp/khmerclipper-merge/trocr
git fetch mac
git fetch easyocr
git fetch trocr
```

- [ ] **Step 2: Merge mac**

```bash
git merge mac/main --allow-unrelated-histories -m "$(cat <<'EOF'
Merge apps/mac history into monorepo.

EOF
)"
```

If the default branch is not `main` (e.g. `KhmerClipper`), use that ref instead:

```bash
git branch -r | grep mac/
```

- [ ] **Step 3: Merge easyocr**

```bash
git merge easyocr/main --allow-unrelated-histories -m "$(cat <<'EOF'
Merge training/easyocr history into monorepo.

EOF
)"
```

- [ ] **Step 4: Merge trocr**

```bash
git merge trocr/main --allow-unrelated-histories -m "$(cat <<'EOF'
Merge training/trocr history into monorepo.

EOF
)"
```

- [ ] **Step 5: Drop scratch remotes**

```bash
git remote remove mac
git remote remove easyocr
git remote remove trocr
git remote -v
```

Expected: no remotes yet (GitHub added in Task 8).

- [ ] **Step 6: Verify tree layout**

```bash
cd /tmp/khmerclipper-merge/mono
test -f apps/mac/README.md && test -f training/easyocr/pyproject.toml && test -f training/trocr/pyproject.toml
find . -name .git -type d
ls -la
```

Expected: three project markers exist; only one `.git` at mono root.

---

### Task 6: Sanity checks (history + ignores)

**Files:**
- None (verification)

**Interfaces:**
- Consumes: merged mono from Task 5
- Produces: pass/fail gate before swapping into workspace

- [ ] **Step 1: History under each prefix**

```bash
cd /tmp/khmerclipper-merge/mono
git log --oneline -- apps/mac/ | head -5
git log --oneline -- training/easyocr/ | head -5
git log --oneline -- training/trocr/ | head -5
git log --follow --oneline -- apps/mac/README.md | head -5
```

Expected: multiple commits per prefix; `--follow` shows history for mac README.

- [ ] **Step 2: Confirm heavy paths are not tracked**

```bash
cd /tmp/khmerclipper-merge/mono
git ls-files 'apps/mac/build/*' | head
git ls-files 'apps/mac/raw_models/*' | head
git ls-files 'apps/mac/Vendor/*' | head -20
git check-ignore -v apps/mac/build apps/mac/raw_models 2>/dev/null || true
```

Expected: `build/` and `raw_models/` not listed; `Vendor/` only allowlisted EasyOCR scripts/khmer paths if present in history — not venvs/weights blobs. If huge binary paths are tracked, stop and fix with a follow-up filter before push.

- [ ] **Step 3: Working tree size sanity**

```bash
du -sh /tmp/khmerclipper-merge/mono
du -sh /tmp/khmerclipper-merge/mono/apps/mac /tmp/khmerclipper-merge/mono/training/*
```

Expected: mono working tree far smaller than the original `mac/` (~6.4G) if build/Vendor copies were not brought as tracked files. (Untracked leftover files from `cp -a` should not exist here because merges bring git trees only.)

---

### Task 7: Minimal path-reference fixes

**Files:**
- Modify only if needed under:
  - `apps/mac/README.md`
  - `apps/mac/Makefile`
  - `training/easyocr/README.md`
  - `training/trocr/README.md`
  - any scripts that hardcode `EasyOCR-KhmerClipper`, `TrOCR-KhmerClipper`, or absolute old paths

**Interfaces:**
- Consumes: merged tree
- Produces: docs/scripts that still make sense from new cwd

- [ ] **Step 1: Search for old folder names**

```bash
cd /tmp/khmerclipper-merge/mono
rg -n 'EasyOCR-KhmerClipper|TrOCR-KhmerClipper|/mac/|workspace/khmerclipper/mac' \
  --glob '!**/.git/**' || true
```

- [ ] **Step 2: Fix only broken references**

Examples of allowed fixes:

- README clone paths → monorepo-relative (`apps/mac`, etc.)
- Docs saying “cd into EasyOCR-KhmerClipper” → `training/easyocr`

Do **not** rename Python packages or change import paths.

- [ ] **Step 3: Commit if any fixes**

```bash
cd /tmp/khmerclipper-merge/mono
git status
# if changes:
git add -A
git commit -m "$(cat <<'EOF'
Fix path references for monorepo layout.

EOF
)"
```

If no matches needing changes, skip the commit.

---

### Task 8: Backup workspace, swap mono in, create private GitHub remote

**Files:**
- Replace workspace contents under `/Users/kim/Sync/workspace/khmerclipper/`
- Create remote via `gh`

**Interfaces:**
- Consumes: verified `/tmp/khmerclipper-merge/mono`
- Produces: live workspace = monorepo; `origin` = private GitHub; `main` pushed

- [ ] **Step 1: Backup current workspace folders**

```bash
mkdir -p /Users/kim/Sync/workspace/khmerclipper-pre-monorepo-backup
rsync -a /Users/kim/Sync/workspace/khmerclipper/mac \
  /Users/kim/Sync/workspace/khmerclipper/EasyOCR-KhmerClipper \
  /Users/kim/Sync/workspace/khmerclipper/TrOCR-KhmerClipper \
  /Users/kim/Sync/workspace/khmerclipper/docs \
  /Users/kim/Sync/workspace/khmerclipper-pre-monorepo-backup/
```

Expected: backup directory populated. Do not delete backup until Task 9 passes.

- [ ] **Step 2: Swap mono into workspace**

Move aside old project dirs, then move mono contents in:

```bash
cd /Users/kim/Sync/workspace/khmerclipper
mv mac EasyOCR-KhmerClipper TrOCR-KhmerClipper /Users/kim/Sync/workspace/khmerclipper-pre-monorepo-backup/ 2>/dev/null || true
# Prefer rsync of mono (including .git) into workspace:
rsync -a --delete --exclude '.DS_Store' /tmp/khmerclipper-merge/mono/ /Users/kim/Sync/workspace/khmerclipper/
cd /Users/kim/Sync/workspace/khmerclipper
git status
ls apps training README.md .gitignore
```

Expected: workspace shows `apps/`, `training/`, root README, single `.git`. Old nested project folders gone from workspace (present in backup).

**Note:** `rsync --delete` removes workspace files not in mono. Confirm backup exists before this step.

- [ ] **Step 3: Create private GitHub repo and push**

```bash
cd /Users/kim/Sync/workspace/khmerclipper
gh repo create khmerclipper --private --source=. --remote=origin --push
```

If name `khmerclipper` is taken:

```bash
gh repo create khmerclipper-app --private --source=. --remote=origin --push
```

If that also fails, stop and ask the user for a name.

- [ ] **Step 4: Verify remote**

```bash
git remote -v
git status
gh repo view --web
```

Expected: `origin` points at the private GitHub repo; `main` up to date with `origin/main`.

---

### Task 9: Smoke check + cleanup notes

**Files:**
- None required (optional local ignore hygiene)

**Interfaces:**
- Consumes: live monorepo on GitHub
- Produces: confirmation app tree is usable; backup retention decision

- [ ] **Step 1: App smoke (structure)**

```bash
cd /Users/kim/Sync/workspace/khmerclipper/apps/mac
test -f project.yml && test -d Sources && test -f Makefile
head -20 README.md
```

Expected: project files present. Full `make run` optional if Xcode tools available; at minimum confirm tree looks correct.

- [ ] **Step 2: Training smoke (structure)**

```bash
test -f /Users/kim/Sync/workspace/khmerclipper/training/easyocr/pyproject.toml
test -f /Users/kim/Sync/workspace/khmerclipper/training/trocr/pyproject.toml
```

- [ ] **Step 3: Backup retention**

Keep `/Users/kim/Sync/workspace/khmerclipper-pre-monorepo-backup/` until you are satisfied. Delete manually later:

```bash
# ONLY after user confirms — do not run automatically:
# rm -rf /Users/kim/Sync/workspace/khmerclipper-pre-monorepo-backup /tmp/khmerclipper-merge
```

---

## Spec coverage self-review

| Spec requirement | Task |
|------------------|------|
| Preserve 3 histories via filter-repo | 3, 5 |
| Layout `apps/mac`, `training/easyocr`, `training/trocr` | 3–5 |
| Commit dirty first | 2 |
| Private GitHub `khmerclipper` | 8 |
| Root README + gitignore | 4 |
| Scratch then swap + backup | 3, 8 |
| Never commit heavy artifacts | 2 (pre-check), 4 (gitignore), 6 (verify) |
| Minimal path fixes | 7 |
| Out of scope (CI, package renames, publish) | not tasked |

## Placeholder / consistency check

- No TBD/TODO placeholders
- Branch name caveat documented for mac (`main` vs `KhmerClipper`)
- Prefix strings identical across tasks: `apps/mac/`, `training/easyocr/`, `training/trocr/`
