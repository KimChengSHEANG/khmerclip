import json

from khmerclip.history import HistoryStore, ScanRecord


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
