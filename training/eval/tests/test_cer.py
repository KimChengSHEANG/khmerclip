from khmerclip_eval.cer import character_error_rate


def test_identical_is_zero():
    assert character_error_rate("សួស្តី", "សួស្តី") == 0.0


def test_total_mismatch_is_one():
    assert character_error_rate("abc", "xyz") == 1.0


def test_empty_reference_empty_pred():
    assert character_error_rate("", "") == 0.0
