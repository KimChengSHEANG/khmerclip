import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from khm_tess import export_gt as eg


def _png(w=40, h=20) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, "PNG")
    return buf.getvalue()


def _parquet(path, rows, text_col="text"):
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "image": pa.array(
                [{"bytes": b, "path": None} for b, _ in rows],
                type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
            ),
            text_col: pa.array([t for _, t in rows], type=pa.string()),
        }
    )
    pq.write_table(table, path)


def test_clean_label():
    assert eg.clean_label("  ក  ខ \n") == "ក  ខ"
    assert eg.clean_label("") is None
    assert eg.clean_label("   ") is None
    assert eg.clean_label(None) is None
    assert eg.clean_label("ក\nខ") is None
    assert eg.clean_label("ក\tខ") is None
    assert eg.clean_label("ក\rខ") is None


def test_load_blocklist(tmp_path):
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps({"id": "a", "image": "a.png", "text": " ក ខ ", "subset": "printed"})
        + "\n\n",
        encoding="utf-8",
    )
    ui = tmp_path / "ui.txt"
    ui.write_text("បើកឯកសារ\n\n", encoding="utf-8")
    assert eg.load_blocklist(labels, ui) == {"ក ខ", "បើកឯកសារ"}


def test_source_files_excludes_test_splits(tmp_path):
    hub = tmp_path / "hub"
    soy = hub / "datasets--SoyVitou--62k-images-khmer-printed-dataset/snapshots/s1"
    for name in ("trainset.parquet", "validset.parquet", "testset.parquet"):
        _parquet(soy / name, [(_png(), name)])
    sok = hub / "datasets--Sokheng--khmer-synthetic-ocr-v1-100k/snapshots/s2/data"
    _parquet(sok / "train-00000-of-00001.parquet", [(_png(), "t")])
    _parquet(sok / "test-00000-of-00001.parquet", [(_png(), "x")])

    assert [p.name for p in eg.source_files(hub, "soyvitou-train")] == ["trainset.parquet"]
    assert [p.name for p in eg.source_files(hub, "soyvitou-valid")] == ["validset.parquet"]
    assert [p.name for p in eg.source_files(hub, "sokheng")] == ["train-00000-of-00001.parquet"]
    with pytest.raises(FileNotFoundError):
        eg.source_files(hub, "hanuman")


def test_iter_rows_stride_sampling(tmp_path):
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    _parquet(a, [(_png(), f"r{i}") for i in range(4)])
    _parquet(b, [(_png(), f"r{i}") for i in range(4, 10)])
    assert [i for i, _, _ in eg.iter_rows([a, b], 3)] == [0, 3, 6]
    assert [t for _, _, t in eg.iter_rows([a, b], None)] == [f"r{i}" for i in range(10)]
    assert len(list(eg.iter_rows([a, b], 50))) == 10


def test_iter_rows_custom_text_column(tmp_path):
    f = tmp_path / "d.parquet"
    _parquet(f, [(_png(), "ក")], text_col="label")
    assert [t for _, _, t in eg.iter_rows([f], None, "label")] == ["ក"]
    assert eg.SOURCES["darayut"][2] == "label"


def test_export_source_writes_pairs_and_stats(tmp_path):
    src = tmp_path / "s.parquet"
    _parquet(
        src,
        [
            (_png(), "ក ខ"),          # kept, has space
            (_png(), "Hello ក"),      # kept, latin + space
            (_png(), "LEAK"),         # blocklisted
            (_png(), "   "),          # empty label
            (b"not an image", "ក"),   # undecodable image
            (_png(3, 3), "ខ"),        # too small
            (None, "គ"),              # null image bytes
        ],
    )
    out = tmp_path / "gt"
    out.mkdir()
    stats = eg.export_source("hanuman", [src], out, {"LEAK"})
    assert stats == {
        "kept": 2,
        "dropped_label": 1,
        "dropped_blocklist": 1,
        "dropped_image": 3,
        "with_space": 2,
        "with_latin": 1,
    }
    assert (out / "hanuman-00000000.gt.txt").read_text(encoding="utf-8") == "ក ខ\n"
    assert Image.open(out / "hanuman-00000001.png").mode == "L"
    assert sorted(p.name for p in out.iterdir()) == [
        "hanuman-00000000.gt.txt",
        "hanuman-00000000.png",
        "hanuman-00000001.gt.txt",
        "hanuman-00000001.png",
    ]


def test_export_flattens_transparent_background_to_white(tmp_path):
    im = Image.new("RGBA", (40, 20), (0, 0, 0, 0))  # fully transparent black
    im.putpixel((5, 5), (0, 0, 0, 255))               # one opaque black "ink" pixel
    buf = io.BytesIO()
    im.save(buf, "PNG")
    src = tmp_path / "s.parquet"
    _parquet(src, [(buf.getvalue(), "ក")])
    out = tmp_path / "gt"
    out.mkdir()
    eg.export_source("sokheng", [src], out, set())
    gray = Image.open(out / "sokheng-00000000.png")
    assert gray.getpixel((0, 0)) == 255   # background white, not black
    assert gray.getpixel((5, 5)) == 0     # ink preserved


def test_export_source_resumes_and_repairs_partial(tmp_path):
    src = tmp_path / "s.parquet"
    _parquet(src, [(_png(), "ក"), (_png(), "ខ")])
    out = tmp_path / "gt"
    out.mkdir()
    eg.export_source("kiteaether", [src], out, set())
    (out / "kiteaether-00000001.gt.txt").unlink()  # simulate crash before gt write
    stats = eg.export_source("kiteaether", [src], out, set())
    assert stats["kept"] == 2
    assert (out / "kiteaether-00000001.gt.txt").read_text(encoding="utf-8") == "ខ\n"


def test_parse_source():
    assert eg.parse_source("mrrtmob:200000") == ("mrrtmob", 200000)
    assert eg.parse_source("hanuman") == ("hanuman", None)
    with pytest.raises(SystemExit):
        eg.parse_source("bogus")
