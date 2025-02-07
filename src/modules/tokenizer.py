import json
import os
from typing import Union

import numpy as np
import torch

from src.utils import hp


class WhisperTokenizerForDiarization:
    def __init__(self):
        self.vocab = self.load_vocab(hp.vocab_path)
        self.prefix_token = [self.vocab["<|startofdiarization|>"], self.vocab["<|zh|>"]]
        self.suffix_token = [self.vocab["<|endofdiarization|>"]]
        self.id2token = {v: k for k, v in self.vocab.items()}

    def load_vocab(self, vocab_path: str) -> dict[str, int]:
        assert os.path.isfile(vocab_path), "vocab file not found"

        with open(vocab_path, "r") as json_obj:
            vocab = json.load(json_obj)

        return vocab

    def add_special_token(self, input_ids: Union[list[int], np.ndarray]):
        if isinstance(input_ids, np.ndarray):
            input_ids = input_ids.tolist()
            return np.array(self.prefix_token + input_ids + self.suffix_token)
        if isinstance(input_ids, list):
            return self.prefix_token + input_ids + self.suffix_token

    def pad(
        self, text: list[torch.Tensor], dynamic_padding: bool = True
    ) -> list[torch.Tensor]:
        max_length = None
        if dynamic_padding:
            max_length = max([t.size(0) for t in text])
        else:
            max_length = hp.max_length

        for i, t in enumerate(text):
            text[i] = torch.cat(
                [
                    t,
                    torch.tensor([self.vocab["<|endofdiarization|>"]] * max_length),
                ],
                dim=0,
            )[:max_length]
        return text

    def shift(self, text: torch.Tensor) -> torch.Tensor:
        """在默认第一个token为起始token的情况下, 将text向右移动一位"""
        text = torch.cat([text[:, :1], text[:, :-1]], dim=1)
        return text
