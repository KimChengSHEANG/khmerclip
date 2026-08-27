from jiwer import cer


def character_error_rate(prediction: str, reference: str) -> float:
    if reference == "" and prediction == "":
        return 0.0
    if reference == "":
        return 1.0
    return float(cer(reference, prediction))
