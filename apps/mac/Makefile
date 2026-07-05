# Stable signing identity keeps the Accessibility (TCC) grant across rebuilds.
SIGN_ID = 
APP      = build/Build/Products/Debug/Netra.app

.PHONY: gen build test run clean trocr-models trocr-khmer easyocr-venv easyocr-models

# Local Khmer checkpoint to (re)build from. Drop a new fine-tune here and
# `make trocr-khmer` reconverts it into Vendor/trocr/khmer. Override the dir
# with `make trocr-khmer KHMER_CKPT=path/to/checkpoint`.
KHMER_CKPT = raw_models/khmer

gen:
	xcodegen generate

# Converts the microsoft/trocr-{small,base,large}-printed checkpoints plus
# the khmer fine-tune to the CoreML models + tokenizer TrOCREngine.swift
# loads (~3.7GB+, not committed -- see .gitignore). Needs:
# pip install torch transformers coremltools. Re-run after pointing
# convert.py/export_tokenizer.py at a different checkpoint (e.g. a newer
# Khmer fine-tune) by passing it as a 3rd argument to those scripts directly.
trocr-models:
	@for size in small base large khmer; do \
		mkdir -p Vendor/trocr/$$size; \
		python3 Vendor/trocr/convert.py $$size Vendor/trocr/$$size; \
		python3 Vendor/trocr/export_tokenizer.py $$size Vendor/trocr/$$size/tokenizer.json; \
	done

# Rebuild only the Khmer model from the local $(KHMER_CKPT) checkpoint into
# Vendor/trocr/khmer. Use after swapping the fine-tune in raw_models/khmer.
trocr-khmer:
	mkdir -p Vendor/trocr/khmer
	python3 Vendor/trocr/convert.py khmer Vendor/trocr/khmer $(KHMER_CKPT)
	python3 Vendor/trocr/export_tokenizer.py khmer Vendor/trocr/khmer/tokenizer.json $(KHMER_CKPT)

# Creates Vendor/easyocr/venv with EasyOCR + PyTorch. Re-run after upgrading
# requirements.txt. The app prefers venv/bin/python3 when present.
easyocr-venv:
	python3 -m venv Vendor/easyocr/venv
	Vendor/easyocr/venv/bin/pip install -r Vendor/easyocr/requirements.txt

# Downloads CRAFT detection weights into Vendor/easyocr/models/. Khmer
# recognition weights (khmer.pth) are installed separately via
# Vendor/easyocr/install_khmer_model.py when you have a fine-tuned model.
easyocr-models: easyocr-venv
	Vendor/easyocr/venv/bin/python3 -c "\
import easyocr; \
easyocr.Reader(['en'], gpu=False, \
model_storage_directory='Vendor/easyocr/models', \
download_enabled=True, verbose=True)"

test: gen
	xcodebuild -project Netra.xcodeproj -scheme NetraTests -configuration Debug \
		-derivedDataPath build \
		CODE_SIGN_STYLE=Manual \
		CODE_SIGN_IDENTITY="$(SIGN_ID)" \
		PROVISIONING_PROFILE_SPECIFIER="" DEVELOPMENT_TEAM="" \
		test

build: gen
	xcodebuild -project Netra.xcodeproj -scheme Netra -configuration Debug \
		-derivedDataPath build \
		CODE_SIGN_STYLE=Manual \
		CODE_SIGN_IDENTITY="$(SIGN_ID)" \
		PROVISIONING_PROFILE_SPECIFIER="" DEVELOPMENT_TEAM="" \
		build

run: build
	@pkill -x Netra 2>/dev/null || true
	@while pgrep -x Netra >/dev/null; do sleep 0.1; done
	@/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$(APP)"
	@sleep 0.5   # let lsregister commit before `open`, else LaunchServices returns -600
	@for i in 1 2 3 4 5; do \
		open "$(APP)" && break || { echo "open failed (LaunchServices not ready), retrying..."; sleep 0.5; }; \
	done

clean:
	rm -rf build Netra.xcodeproj
