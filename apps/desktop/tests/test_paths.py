from khmerclipper import paths


def test_resource_dir_contains_bundled_font_and_icon():
    root = paths.resource_dir()
    assert (root / "fonts" / "NotoSansKhmer-Regular.ttf").is_file()
    assert (root / "icon.png").is_file()


def test_atomic_write_text_writes_utf8_and_leaves_no_temp(tmp_path):
    target = tmp_path / "sub" / "f.json"
    paths.atomic_write_text(target, "ខ្មែរ")
    assert target.read_bytes().decode("utf-8") == "ខ្មែរ"
    assert [p.name for p in target.parent.iterdir()] == ["f.json"]
