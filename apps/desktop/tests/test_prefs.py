import json

from khmerclipper.prefs import Prefs


def test_missing_file_gives_defaults(tmp_path):
    prefs = Prefs.load(tmp_path / "prefs.json")
    assert prefs.engine == "trocr"
    assert prefs.hotkey == "<ctrl>+<shift>+2"
    assert prefs.auto_copy is True


def test_save_then_load_roundtrips(tmp_path):
    path = tmp_path / "prefs.json"
    prefs = Prefs.load(path)
    prefs.engine = "tesseract"
    prefs.auto_copy = False
    prefs.last_update_check = 123.5
    prefs.save()
    again = Prefs.load(path)
    assert (again.engine, again.auto_copy, again.last_update_check) == ("tesseract", False, 123.5)
    assert "path" not in json.loads(path.read_text(encoding="utf-8"))


def test_corrupt_json_gives_defaults(tmp_path):
    path = tmp_path / "prefs.json"
    path.write_text("{not json", encoding="utf-8")
    assert Prefs.load(path).engine == "trocr"


def test_wrong_typed_fields_fall_back_individually(tmp_path):
    path = tmp_path / "prefs.json"
    path.write_text(json.dumps({"auto_copy": "yes", "show_result": False, "last_update_check": 7,
                                "check_updates": 1, "engine": "easyocr", "bogus": 1}), encoding="utf-8")
    prefs = Prefs.load(path)
    assert prefs.auto_copy is True            # "yes" is not a bool -> default
    assert prefs.show_result is False         # valid -> kept
    assert prefs.last_update_check == 7.0     # int accepted for float
    assert prefs.check_updates is True        # 1 is not a bool -> default
    assert prefs.engine == "trocr"            # unknown engine -> default


def test_keep_captures_on_for_new_installs_only(tmp_path):
    path = tmp_path / "prefs.json"
    assert Prefs.load(path).keep_captures is True  # new install: no prefs file yet
    path.write_text(json.dumps({"engine": "trocr"}), encoding="utf-8")
    assert Prefs.load(path).keep_captures is False  # existing install that never set it
    path.write_text("{not json", encoding="utf-8")
    assert Prefs.load(path).keep_captures is False  # unreadable, but the app has run before
    path.write_text(json.dumps({"keep_captures": True}), encoding="utf-8")
    assert Prefs.load(path).keep_captures is True  # an explicit choice wins
