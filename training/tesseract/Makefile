# Tesseract khm fine-tune. Run on the training server; see README.md.
# All generated data lives in WORK (outside ~/Sync — Syncthing must not see it).
WORK    ?= $(HOME)/khm-tesstrain
HF_HUB  ?= $(HOME)/.cache/huggingface/hub
IMAGE   ?= khm-tesstrain
MODEL   ?= khm_ft
CPUS    ?= 8
JOBS    ?= 10
EPOCHS  ?= 3
MAX_ITERATIONS ?=
SOURCES ?=
DETACH  ?= -d

OUT   = $(WORK)/data/$(MODEL)
UV    = UV_PROJECT_ENVIRONMENT=$(WORK)/.venv uv run
DOCKER_ARGS = --cpus $(CPUS) --user $$(id -u):$$(id -g) -e HOME=/tmp -e OMP_THREAD_LIMIT=$(CPUS) -v $(WORK):/work
RUN   = docker run --rm $(DOCKER_ARGS) $(IMAGE)
TT    = make -C /opt/tesstrain DATA_DIR=/work/data TESSDATA=/opt/tessdata_best START_MODEL=khm MODEL_NAME=$(MODEL) PSM=13
ITERS = $(if $(MAX_ITERATIONS),MAX_ITERATIONS=$(MAX_ITERATIONS),EPOCHS=$(EPOCHS))

.PHONY: image check download gt dicts lists train logs evaluate test

image:
	docker build -t $(IMAGE) .

check:
	docker run --rm $(IMAGE) sh -c 'set -e; tesseract --version 2>&1 | head -1; tesseract --list-langs; \
	  ls /usr/share/tesseract-ocr/5/tessdata/configs/lstm.train; \
	  command -v lstmtraining lstmeval combine_tessdata combine_lang_model unicharset_extractor merge_unicharsets dawg2wordlist; \
	  git -C /opt/tesstrain rev-parse HEAD; sha256sum /opt/tessdata_best/khm.traineddata; \
	  combine_tessdata -d /opt/tessdata_best/khm.traineddata'

# Sources not yet in the hub cache (the other five are cached already).
download:
	$(UV) --with "huggingface_hub>=0.34" hf download Darayut/khmer-document-synthetic-low-res --repo-type dataset --include "data/*"
	$(UV) --with "huggingface_hub>=0.34" hf download KimkosalYon/khmer-ocr-200k-siemreap-arial --repo-type dataset --include "data/*"

gt:
	mkdir -p $(WORK)/data/$(MODEL)-ground-truth
	$(UV) khm-export-gt --hub $(HF_HUB) --out $(WORK)/data/$(MODEL)-ground-truth $(foreach s,$(SOURCES),--source $(s))

# Keep khm's dictionary: tesstrain builds the proto model from these three files if present.
dicts:
	mkdir -p $(OUT)
	$(RUN) sh -c 'set -e; mkdir -p /tmp/khm; combine_tessdata -u /opt/tessdata_best/khm.traineddata /tmp/khm/khm.; \
	  for d in word:wordlist punc:punc number:numbers; do \
	    dawg=/tmp/khm/khm.lstm-$${d%%:*}-dawg; \
	    if [ -f $$dawg ]; then dawg2wordlist /tmp/khm/khm.lstm-unicharset $$dawg /work/data/$(MODEL)/$(MODEL).$${d##*:}; \
	    else echo "skip: $$dawg not in start model"; fi; \
	  done; wc -l /work/data/$(MODEL)/$(MODEL).* || true'

lists:
	$(RUN) $(TT) -j $(JOBS) lists unicharset
	$(UV) khm-split-lists $(OUT)

# Re-running resumes: tesstrain's training target depends on phony targets, so lstmtraining always
# runs, and lstmtraining restores <model>_checkpoint when it exists.
train:
	-docker rm $(MODEL)-train
	docker run $(DETACH) --name $(MODEL)-train $(DOCKER_ARGS) $(IMAGE) $(TT) training $(ITERS)

logs:
	tail -f $(OUT)/training.log

# BCER on list.eval: start model first, then every checkpoint (lowest wins).
evaluate:
	$(RUN) sh -c 'lstmeval --model /opt/tessdata_best/khm.traineddata --eval_listfile /work/data/$(MODEL)/list.eval 2>&1 | grep "^BCER eval" | sed "s/^/START /"'
	$(RUN) $(TT) -j $(JOBS) traineddata evaluation
	grep -H "^BCER eval" $(OUT)/eval/*.eval.log | sort -t= -k2 -g | head -5

test:
	$(UV) --extra dev pytest -q
