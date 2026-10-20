"""Fetch pinned tessdata_fast models into resources/tesseract/tessdata.

Same files (and hashes) the mac app vendors. Run from apps/desktop:
    .venv/bin/python tools/fetch_tessdata.py
"""

from khmerclipper.download import RemoteFile, download
from khmerclipper.ocr.tesseract import tessdata_dir

BASE = "https://github.com/tesseract-ocr/tessdata_fast/raw/4.1.0/"
FILES = [
    RemoteFile("khm.traineddata", BASE + "khm.traineddata",
               "47f110575341b322052f3becbefee61a3ecf1ef549352b5d4d33d28afe30d099", 1446926),
    RemoteFile("eng.traineddata", BASE + "eng.traineddata",
               "7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2", 4113088),
]

if __name__ == "__main__":
    download(FILES, tessdata_dir())
    print("tessdata ready:", tessdata_dir())
