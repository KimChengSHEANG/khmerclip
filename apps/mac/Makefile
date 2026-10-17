# Stable signing identity keeps the Accessibility (TCC) grant across rebuilds.
SIGN_ID = 
APP      = build/Build/Products/Debug/KhmerClip.app

.PHONY: gen build test run clean trocr-models trocr-khmer trocr-test ocrbench

# Local Khmer checkpoint to (re)build from. Drop a new fine-tune here and
# `make trocr-khmer` reconverts it into Vendor/trocr/khmer. Override the dir
# with `make trocr-khmer KHMER_CKPT=path/to/checkpoint`.
KHMER_CKPT = raw_models/TrOCR-khmer

# Conversion-only Python env (torch/transformers/coremltools). Kept outside
# Vendor/trocr because that whole folder is bundled into the app.
TROCR_VENV = .trocr-venv
TROCR_PY   = $(TROCR_VENV)/bin/python3

# build/test reconvert the Khmer model automatically when the checkpoint's
# weights are newer than the converted output. Skipped if no checkpoint exists.
KHMER_SRC   = $(wildcard $(KHMER_CKPT)/model.safetensors)
KHMER_MODEL = $(if $(KHMER_SRC),Vendor/trocr/khmer/tokenizer.json)

gen:
	xcodegen generate

# Converts the microsoft/trocr-{small,base,large}-printed checkpoints plus
# the khmer fine-tune to the CoreML models + tokenizer TrOCREngine.swift
# loads (~3.7GB+, not committed -- see .gitignore). Needs:
# pip install torch transformers coremltools. Re-run after pointing
# convert.py/export_tokenizer.py at a different checkpoint (e.g. a newer
# Khmer fine-tune) by passing it as a 3rd argument to those scripts directly.
trocr-models: $(TROCR_PY)
	@for size in small base large khmer; do \
		mkdir -p Vendor/trocr/$$size; \
		$(TROCR_PY) tools/trocr/convert.py $$size Vendor/trocr/$$size; \
		$(TROCR_PY) tools/trocr/export_tokenizer.py $$size Vendor/trocr/$$size/tokenizer.json; \
	done

# Rebuild only the Khmer model from the local $(KHMER_CKPT) checkpoint into
# Vendor/trocr/khmer. Use after swapping the fine-tune in raw_models/TrOCR-khmer.
$(TROCR_PY):
	python3 -m venv $(TROCR_VENV)
	$(TROCR_PY) -m pip install torch==2.7.0 "transformers==4.57.*" coremltools pillow pytest jiwer

# Converter unit tests (tiny random models; no checkpoint or network needed).
trocr-test: | $(TROCR_PY)
	$(TROCR_PY) -m pytest tools/trocr/tests -v

# tokenizer.json is written last, so it doubles as the up-to-date stamp.
trocr-khmer:
	rm -f Vendor/trocr/khmer/tokenizer.json
	$(MAKE) Vendor/trocr/khmer/tokenizer.json

# Converter changes (e.g. the KV-cache decoder export) also trigger a rebuild.
Vendor/trocr/khmer/tokenizer.json: $(KHMER_SRC) $(if $(KHMER_SRC),$(wildcard tools/trocr/*.py)) | $(TROCR_PY)
	mkdir -p Vendor/trocr/khmer
	$(TROCR_PY) tools/trocr/convert.py khmer Vendor/trocr/khmer $(KHMER_CKPT)
	$(TROCR_PY) tools/trocr/export_tokenizer.py khmer $@ $(KHMER_CKPT)

# Release OCRBench CLI for the eval harness (training/eval/scripts/bench.py
# --trocr-coreml-bin build/Build/Products/Release/OCRBench). Tool targets
# need an ad-hoc identity; an empty one fails to sign.
ocrbench: gen $(KHMER_MODEL)
	xcodebuild -project KhmerClip.xcodeproj -scheme OCRBench -configuration Release \
		-derivedDataPath build CODE_SIGN_IDENTITY="-" CODE_SIGNING_REQUIRED=NO build

test: gen $(KHMER_MODEL)
	xcodebuild -project KhmerClip.xcodeproj -scheme KhmerClipTests -configuration Debug \
		-derivedDataPath build \
		CODE_SIGN_STYLE=Manual \
		CODE_SIGN_IDENTITY="$(SIGN_ID)" \
		PROVISIONING_PROFILE_SPECIFIER="" DEVELOPMENT_TEAM="" \
		test

build: gen $(KHMER_MODEL)
	xcodebuild -project KhmerClip.xcodeproj -scheme KhmerClip -configuration Debug \
		-derivedDataPath build \
		CODE_SIGN_STYLE=Manual \
		CODE_SIGN_IDENTITY="$(SIGN_ID)" \
		PROVISIONING_PROFILE_SPECIFIER="" DEVELOPMENT_TEAM="" \
		build

run: build
	@pkill -x KhmerClip 2>/dev/null || true
	@while pgrep -x KhmerClip >/dev/null; do sleep 0.1; done
	@/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$(APP)"
	@sleep 0.5   # let lsregister commit before `open`, else LaunchServices returns -600
	@for i in 1 2 3 4 5; do \
		open "$(APP)" && break || { echo "open failed (LaunchServices not ready), retrying..."; sleep 0.5; }; \
	done

clean:
	rm -rf build KhmerClip.xcodeproj
