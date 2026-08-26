"""Label converters and helpers.

Adapted from Clova AI's deep-text-recognition-benchmark (Apache-2.0).
"""
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CTCLabelConverter(object):
    """Convert between text-label and text-index for CTC."""

    def __init__(self, character):
        dict_character = list(character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            # index 0 is reserved for the CTC blank token.
            self.dict[char] = i + 1
        self.character = ["[blank]"] + dict_character  # blank at index 0

    def encode(self, text, batch_max_length=25):
        length = []
        indexed = []
        for s in text:
            indices = [self.dict[c] for c in s if c in self.dict]
            length.append(len(indices))
            indexed.extend(indices)
        text = torch.IntTensor(indexed)
        return text.to(device), torch.IntTensor(length).to(device)

    def decode(self, text_index, length):
        texts = []
        index = 0
        for l in length:
            t = text_index[index : index + l]
            char_list = []
            for i in range(l):
                # remove repeated chars and blanks (CTC decoding rule)
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(self.character[t[i]])
            texts.append("".join(char_list))
            index += l
        return texts


class CTCLabelConverterForBaiduWarpctc(object):
    """CTC converter that returns a flat concatenated label tensor."""

    def __init__(self, character):
        dict_character = list(character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i + 1
        self.character = ["[blank]"] + dict_character

    def encode(self, text, batch_max_length=25):
        length = [len(s) for s in text]
        text = "".join(text)
        text = [self.dict[char] for char in text]
        return torch.IntTensor(text), torch.IntTensor(length)

    def decode(self, text_index, length):
        texts = []
        index = 0
        for l in length:
            t = text_index[index : index + l]
            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(self.character[t[i]])
            texts.append("".join(char_list))
            index += l
        return texts


class AttnLabelConverter(object):
    """Convert between text-label and text-index for attention decoder."""

    def __init__(self, character):
        # [GO] for start token, [s] for end token.
        list_token = ["[GO]", "[s]"]
        list_character = list(character)
        self.character = list_token + list_character
        self.dict = {}
        for i, char in enumerate(self.character):
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        length = [len(s) + 1 for s in text]  # +1 for [s] at end
        batch_max_length += 1
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            txt = list(t)
            txt.append("[s]")
            txt = [self.dict[char] for char in txt if char in self.dict]
            batch_text[i][1 : 1 + len(txt)] = torch.LongTensor(txt)
        return batch_text.to(device), torch.IntTensor(length).to(device)

    def decode(self, text_index, length):
        texts = []
        for index, l in enumerate(length):
            text = "".join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


class Averager(object):
    """Compute the average for `torch.Tensor`."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res
