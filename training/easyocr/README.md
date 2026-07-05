# Khmer EasyOCR — Training & Evaluation

Train and evaluate an **EasyOCR-compatible** text-recognition model for **Khmer**
using the [SoyVitou/62k-images-khmer-printed-dataset](https://huggingface.co/datasets/SoyVitou/62k-images-khmer-printed-dataset)
(62,680 printed Khmer/English line images, columns `image` + `text`).

The recognizer architecture and training loop are a self-contained port of the
trainer that ships with [EasyOCR](https://github.com/JaidedAI/EasyOCR) (itself
based on Clova AI's *deep-text-recognition-benchmark*). Weights produced here are
loadable by the `easyocr` package as a **custom recognition network**.

## Project layout

```
EasyOCR-KhmerClip/
├── pyproject.toml            installable package + console scripts
├── requirements.txt
├── README.md
├── configs/
│   └── khmer.yaml            training / model configuration
└── src/
    └── khmer_ocr/            main package
        ├── config.py         config + charset (de)serialisation
        ├── data.py           Hugging Face dataset -> LMDB + charset
        ├── evaluate.py       CER / WER / accuracy / normalized edit distance
        ├── packaging.py      export a checkpoint for the easyocr package
        ├── cli/              command-line entry points
        │   ├── prepare_data.py
        │   ├── train.py
        │   ├── evaluate.py
        │   └── package_model.py
        └── trainer/          vendored EasyOCR-style recognizer
            ├── model.py      Trans -> Feat -> Seq -> Pred model
            ├── engine.py     training + validation loop
            ├── dataset.py    LMDB dataset, resize+pad collate, balanced sampler
            ├── utils.py      CTC / Attn label converters
            └── modules/      transformation / feature / sequence / prediction
```

## Pipeline

```
prepare  ->  train  ->  evaluate  ->  package (optional)
(HF->LMDB)  (train)     (CER/WER)     (use in easyocr)
```

## 1. Setup

Tuned for the local `khmerclip` pyenv environment (Python 3.12, torch + CUDA).
Install the package (editable) so the CLIs and imports work from anywhere:

```bash
pip install -e .
# or just the deps: pip install -r requirements.txt
```

This exposes four console commands: `khmer-ocr-prepare`, `khmer-ocr-train`,
`khmer-ocr-eval`, `khmer-ocr-package`. Each is also runnable as
`python -m khmer_ocr.cli.<name>`.

## 2. Prepare the data

Downloads the parquet splits from the Hub, converts them to LMDB, and writes the
character set discovered in the labels.

```bash
khmer-ocr-prepare                     # full dataset -> data/lmdb/{train,val,test}
khmer-ocr-prepare --max-samples 2000  # quick subset for a smoke test
```

Outputs: `data/lmdb/{train,val,test}` and `data/charset.txt` (one character per
line; includes the meaningful space character).

## 3. Train

```bash
khmer-ocr-train --config configs/khmer.yaml
```

Handy overrides (no need to edit the YAML):

```bash
khmer-ocr-train --num_iter 300 --batch_size 32 --valInterval 100
khmer-ocr-train --amp                                              # mixed precision
khmer-ocr-train --saved_model saved_models/khmer_g2/best_norm_ED.pth  # resume
```

Checkpoints and logs land in `saved_models/<experiment_name>/`
(`best_accuracy.pth`, `best_norm_ED.pth`, `iter_*.pth`, `log_train.txt`).
Default architecture: `None-VGG-BiLSTM-CTC` (EasyOCR "generation 2"), `imgH=64`,
`imgW=600`, `batch_max_length=120`.

## 4. Evaluate

```bash
khmer-ocr-eval --config configs/khmer.yaml \
    --saved_model saved_models/khmer_g2/best_norm_ED.pth \
    --eval_data data/lmdb/test --save_preds preds_test.csv
```

Reports **exact-match accuracy**, **CER**, **WER** (via `jiwer`) and the
**normalized edit distance** used by the trainer. `preds_test.csv` holds
per-line `ground_truth, prediction, confidence, correct`.

## 5. (Optional) Use inside the `easyocr` package

```bash
pip install easyocr        # optional; may change your torch version
khmer-ocr-package --config configs/khmer.yaml \
    --saved_model saved_models/khmer_g2/best_norm_ED.pth --name khmer_g2
```

```python
import easyocr
reader = easyocr.Reader(['en'], recog_network='khmer_g2')
print(reader.readtext('line.jpg', detail=0))
```

## Notes

- Khmer uses combining characters and a meaningful space character; the charset
  is built directly from the dataset labels so all of these are covered.
- CTC decoding is greedy by default (`decode: greedy` in the config).
- For a stronger model try `FeatureExtraction: ResNet` and/or
  `Transformation: TPS` in the config (slower, more accurate).
