"""Sequence-modeling stage (BiLSTM).

Adapted from Clova AI's deep-text-recognition-benchmark (Apache-2.0).
"""
import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """input : visual feature [batch_size x T x input_size]."""
        try:
            self.rnn.flatten_parameters()
        except Exception:
            pass
        recurrent, _ = self.rnn(input)  # [b, T, 2*hidden]
        output = self.linear(recurrent)  # [b, T, output_size]
        return output
