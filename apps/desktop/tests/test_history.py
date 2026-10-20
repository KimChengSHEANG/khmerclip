import json

from PIL import Image

from khmerclipper.history import HistoryStore, ScanRecord


def test_add_puts_newest_first_and_persists(tmp_path):
    path = tmp_path / "history.json"
    store = HistoryStore(path)
    store.add(ScanRecord("first"))
    store.add(ScanRecord("ទីពីរ"))
    assert [r.text for r in store.records] == ["ទីពីរ", "first"]
    assert [r.text for r in HistoryStore(path).records] == ["ទីពីរ", "first"]
    assert "ទីពីរ" in path.read_bytes().decode("utf-8")  # raw UTF-8, not \u escapes


def test_caps_at_max_records(tmp_path):
    store = HistoryStore(tmp_path / "h.json")
    for i in range(HistoryStore.MAX_RECORDS + 5):
        store.add(ScanRecord(str(i)))
    assert len(store.records) == HistoryStore.MAX_RECORDS
    assert store.records[0].text == str(HistoryStore.MAX_RECORDS + 4)


def test_remove_and_clear(tmp_path):
    store = HistoryStore(tmp_path / "h.json")
    keep, drop = ScanRecord("keep"), ScanRecord("drop")
    store.add(keep)
    store.add(drop)
    store.remove(drop.id)
    assert [r.text for r in store.records] == ["keep"]
    store.clear()
    assert HistoryStore(tmp_path / "h.json").records == []


def test_corrupt_file_is_quarantined_not_lost(tmp_path):
    path = tmp_path / "history.json"
    path.write_text("[{broken", encoding="utf-8")
    store = HistoryStore(path)
    assert store.records == []
    assert (tmp_path / "history.json.bak").read_text(encoding="utf-8") == "[{broken"


def test_wrong_shape_is_quarantined(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(json.dumps({"text": "x"}), encoding="utf-8")
    assert HistoryStore(path).records == []
    assert (tmp_path / "history.json.bak").exists()


def test_captures_are_kept_flagged_and_cleaned_up(tmp_path):

    store = HistoryStore(tmp_path / "h.json")
    plain, kept = ScanRecord("no image"), ScanRecord("ក")
    store.add(plain)
    store.add(kept, image=Image.new("RGB", (8, 4), "white"), meta={"ocr_text": "ក", "engine": "trocr"})
    assert not store.has_capture(plain.id) and store.has_capture(kept.id)

    store.flag(kept.id)
    assert store.is_flagged(kept.id)
    assert json.loads((tmp_path / "feedback" / f"{kept.id}.json").read_text(encoding="utf-8"))["ocr_text"] == "ក"

    store.remove(kept.id)
    assert not store.has_capture(kept.id)
    assert store.is_flagged(kept.id)  # feedback survives the history

    for i in range(HistoryStore.MAX_RECORDS + 1):  # the oldest capture falls off with its record
        store.add(ScanRecord(str(i)), image=Image.new("RGB", (2, 2)) if i == 0 else None)
    assert list((tmp_path / "captures").glob("*.png")) == []

    store.add(ScanRecord("x"), image=Image.new("RGB", (2, 2)))
    store.clear()
    assert not (tmp_path / "captures").exists()


def test_flag_stores_correction(tmp_path):
    store = HistoryStore(tmp_path / "h.json"); rec = ScanRecord("ក")
    store.add(rec, image=Image.new("RGB", (8, 4)), meta={"ocr_text": "ក"})
    store.flag(rec.id, "ខ")
    saved = json.loads((tmp_path / "feedback" / f"{rec.id}.json").read_text(encoding="utf-8"))
    assert saved == {"ocr_text": "ក", "correction": "ខ"}
    assert store.capture_meta(rec.id) == {"ocr_text": "ក"}
    assert store.capture_meta("missing") == {}
