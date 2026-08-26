"""Export a trained checkpoint so the official `easyocr` package can load it as a
custom recognition network."""
import os
import shutil

import yaml

from .config import load_charset

USER_NETWORK_TEMPLATE = '''"""Auto-generated EasyOCR custom recognition network."""
import sys
from types import SimpleNamespace

from khmer_ocr.trainer.model import Model as _BaseModel

# Fallback for a non-installed checkout: add the src/ dir to sys.path.
_SRC = r"{src_path}"
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


class Model(_BaseModel):
    """EasyOCR instantiates this as Model(input_channel, output_channel,
    hidden_size, num_class). We map those onto the vendored architecture."""

    def __init__(self, input_channel, output_channel, hidden_size, num_class, **kwargs):
        opt = SimpleNamespace(
            Transformation="{transformation}",
            FeatureExtraction="{feature}",
            SequenceModeling="{sequence}",
            Prediction="{prediction}",
            num_fiducial={num_fiducial},
            imgH={imgH},
            imgW={imgW},
            input_channel=input_channel,
            output_channel=output_channel,
            hidden_size=hidden_size,
            num_class=num_class,
            batch_max_length={batch_max_length},
        )
        super().__init__(opt)

    def forward(self, input, text):
        return super().forward(input, text)
'''


def package_model(opt, saved_model, name, easyocr_dir, src_path):
    """Write model/<name>.pth, user_network/<name>.py and <name>.yaml."""
    character = load_charset(opt.character_file)

    model_dir = os.path.join(easyocr_dir, "model")
    net_dir = os.path.join(easyocr_dir, "user_network")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(net_dir, exist_ok=True)

    dst_pth = os.path.join(model_dir, f"{name}.pth")
    shutil.copyfile(saved_model, dst_pth)

    py = USER_NETWORK_TEMPLATE.format(
        src_path=src_path,
        transformation=opt.Transformation,
        feature=opt.FeatureExtraction,
        sequence=opt.SequenceModeling,
        prediction=opt.Prediction,
        num_fiducial=opt.num_fiducial,
        imgH=opt.imgH,
        imgW=opt.imgW,
        batch_max_length=opt.batch_max_length,
    )
    py_path = os.path.join(net_dir, f"{name}.py")
    with open(py_path, "w", encoding="utf-8") as f:
        f.write(py)

    cfg = {
        "network_params": {
            "input_channel": opt.input_channel,
            "output_channel": opt.output_channel,
            "hidden_size": opt.hidden_size,
        },
        "imgH": opt.imgH,
        "lang_list": ["en"],  # 'km' has no built-in detector; use the latin detector
        "character_list": character,
    }
    yaml_path = os.path.join(net_dir, f"{name}.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    return dst_pth, py_path, yaml_path
