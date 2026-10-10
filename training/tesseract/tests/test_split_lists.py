import pytest

from khm_tess.split_lists import main, split


def test_split_keeps_valid_out_of_train():
    paths = [
        "/work/data/khm_ft-ground-truth/hanuman-00000001.lstmf",
        "/work/data/khm_ft-ground-truth/soyvitou-valid-00000003.lstmf",
        "/work/data/khm_ft-ground-truth/soyvitou-train-00000002.lstmf",
    ]
    train, eval_ = split(paths)
    assert train == [paths[0], paths[2]]
    assert eval_ == [paths[1]]


def test_split_errors_when_side_empty():
    with pytest.raises(ValueError, match="eval"):
        split(["/x/hanuman-00000001.lstmf"])
    with pytest.raises(ValueError, match="train"):
        split(["/x/soyvitou-valid-00000001.lstmf"])


def test_main_overwrites_lists(tmp_path, monkeypatch):
    (tmp_path / "all-lstmf").write_text("/g/hanuman-1.lstmf\n/g/soyvitou-valid-2.lstmf\n\n")
    (tmp_path / "list.train").write_text("stale\n")
    monkeypatch.setattr("sys.argv", ["khm-split-lists", str(tmp_path)])
    main()
    assert (tmp_path / "list.train").read_text() == "/g/hanuman-1.lstmf\n"
    assert (tmp_path / "list.eval").read_text() == "/g/soyvitou-valid-2.lstmf\n"
