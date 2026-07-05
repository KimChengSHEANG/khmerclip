"""Recognition model = Transformation -> FeatureExtraction -> Sequence -> Prediction.

Adapted from Clova AI's deep-text-recognition-benchmark (Apache-2.0). The layer
naming is kept identical so checkpoints are loadable by the `easyocr` package.
"""
import torch.nn as nn

from .modules.transformation import TPS_SpatialTransformerNetwork
from .modules.feature_extraction import (
    VGG_FeatureExtractor,
    RCNN_FeatureExtractor,
    ResNet_FeatureExtractor,
)
from .modules.sequence_modeling import BidirectionalLSTM
from .modules.prediction import Attention


class Model(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.stages = {
            "Trans": opt.Transformation,
            "Feat": opt.FeatureExtraction,
            "Seq": opt.SequenceModeling,
            "Pred": opt.Prediction,
        }

        # Transformation
        if opt.Transformation == "TPS":
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial,
                I_size=(opt.imgH, opt.imgW),
                I_r_size=(opt.imgH, opt.imgW),
                I_channel_num=opt.input_channel,
            )
        else:
            print("No Transformation module specified")

        # FeatureExtraction
        if opt.FeatureExtraction == "VGG":
            self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == "RCNN":
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == "ResNet":
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        else:
            raise Exception("No FeatureExtraction module specified")
        self.FeatureExtraction_output = opt.output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        # Sequence modeling
        if opt.SequenceModeling == "BiLSTM":
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size),
            )
            self.SequenceModeling_output = opt.hidden_size
        else:
            print("No SequenceModeling module specified")
            self.SequenceModeling_output = self.FeatureExtraction_output

        # Prediction
        if opt.Prediction == "CTC":
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == "Attn":
            self.Prediction = Attention(
                self.SequenceModeling_output, opt.hidden_size, opt.num_class
            )
        else:
            raise Exception("Prediction is neither CTC or Attn")

    def forward(self, input, text, is_train=True):
        if self.stages["Trans"] == "TPS":
            input = self.Transformation(input)

        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
        visual_feature = visual_feature.squeeze(3)

        if self.stages["Seq"] == "BiLSTM":
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature

        if self.stages["Pred"] == "CTC":
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(
                contextual_feature.contiguous(),
                text,
                is_train,
                batch_max_length=self.opt.batch_max_length,
            )
        return prediction
