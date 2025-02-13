import tempfile

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import WhisperFeatureExtractor

from src.utils import AudioParser, MixDataPipe, hp

from .model import WhisperTokenizerForDiarization


class SpearkerDataset(Dataset):
    def __init__(self, split: str):
        self.mix_data_pipe = MixDataPipe()
        self.audio_parser = AudioParser()
        self.tokenizer = WhisperTokenizerForDiarization()

        self.processor = WhisperFeatureExtractor.from_pretrained(hp.model_name)
        self.mixed_file_list = self._get_mixed_file_list(split)

    def __len__(self) -> int:
        return len(self.mixed_file_list)

    def __getitem__(self, idx: int):
        _mixed_file: list[str] = self.mixed_file_list[idx]

        mixed_file = self._get_seg_tensor_from_list(_mixed_file)
        labels = self._get_seg_labels_from_list(_mixed_file)
        return mixed_file, labels

    def _get_mixed_file_list(self, split: str) -> list[list[str]]:
        mixed_file_list = self.mix_data_pipe.remix(hp.split2path[split])
        return mixed_file_list

    def _get_seg_tensor_from_list(self, audio_path_list: list[str]) -> torch.Tensor:
        with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as temp_file:
            params, nframes = self.audio_parser.merge(audio_path_list)
            self.audio_parser.write(temp_file.name, params, nframes)
            mel = self.processor(
                self.audio_parser.load_audio(temp_file.name), return_tensors="pt"
            )
        return mel.input_features

    def _get_seg_labels_from_list(self, audio_path_list: list[str]) -> torch.Tensor:
        _label = np.arange(len(audio_path_list)) + hp.speaker_idx_offset
        _audio_times = list()
        for audio_path in audio_path_list:
            params, _ = self.audio_parser.read(audio_path)
            _audio_times.append(self.audio_parser.get_audio_time(params))

        _label = np.repeat(_label, _audio_times)

        _label = self.tokenizer.add_special_token(_label)
        return torch.tensor(_label)


def collate_fn(batch):
    tokenizer = WhisperTokenizerForDiarization()
    input_features, labels = [], []
    for item in batch:
        input_features.append(item[0])
        labels.append(item[1])

    labels = tokenizer.pad(labels)

    return torch.concat(input_features), torch.stack(labels)


def get_loader(split, batch_size, shuffle=False):
    assert split in [
        "train",
        "test",
        "valid",
    ], f"Only support [train | test | valid]"

    if split == "train":
        dataset = SpearkerDataset(split)
    elif split == "test":
        dataset = SpearkerDataset(split)
    elif split == "valid":
        dataset = SpearkerDataset(split)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size,
        shuffle,
        collate_fn=collate_fn,
    )
