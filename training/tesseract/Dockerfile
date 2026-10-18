# Tesseract LSTM fine-tune toolchain (CPU only). Pins: tesstrain commit + tessdata_best khm SHA-256.
FROM ubuntu:24.04

ARG TESSTRAIN_REF=405346a3a67d8e4e049341d1da6a4b752e0b8351
ARG TESSDATA_BEST_REF=9ddc24e750eec0994223a9edc3fcb434a2244f3b
ARG KHM_BEST_SHA256=3d2d975e67a319bf1cd67626b12fdebc1e417a4ebef9c9afee01d9927b512815

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      tesseract-ocr make git python3 python3-pil wget ca-certificates bc \
 && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/tesseract-ocr/tesstrain /opt/tesstrain \
 && git -C /opt/tesstrain checkout "$TESSTRAIN_REF"

RUN mkdir -p /opt/tessdata_best \
 && wget -qO /opt/tessdata_best/khm.traineddata \
      "https://github.com/tesseract-ocr/tessdata_best/raw/${TESSDATA_BEST_REF}/khm.traineddata" \
 && echo "${KHM_BEST_SHA256}  /opt/tessdata_best/khm.traineddata" | sha256sum -c -

# Non-root by default; the Makefile also passes --user $(id -u):$(id -g).
USER ubuntu
