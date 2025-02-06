import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.utils import AudioParser, MixDataPipe, hp


class SpearkerDataset(Dataset):
    def __init__(self, split: str):
        self.mix_data_pipe = MixDataPipe()
        self.audio_parser = AudioParser()
        self.mixed_file_list = self._get_mixed_file_list(split)

    def __len__(self) -> int:
        return len(self.mixed_file_list)

    def __getitem__(self, idx: int):
        _mixed_file: list[str] = self.mixed_file_list[idx]
        # _mixed_id: list[int] = list(range(len(_mixed_file)))

        mixed_file = self._get_seg_tensor_from_list(_mixed_file)
        labels = self._get_seg_labels_from_list(_mixed_file)
        return mixed_file, labels

    def _get_mixed_file_list(self, split: str) -> list[list[str]]:
        mixed_file_list = self.mix_data_pipe.remix(hp.split2path[split])
        return mixed_file_list

    def _get_percent(self, input_path: list[str]) -> np.ndarray:
        _percent = list()
        for audio_path in input_path:
            params, _ = self.audio_parser.read(audio_path)
            _percent.append(params.nframes)

        percent: np.ndarray = (np.array(_percent) / np.sum(_percent)) * hp.logits_dim
        percent = percent.round()
        diff = hp.logits_dim - np.sum(percent)
        if diff != 0:
            percent[np.argmax(percent)] += diff

        assert np.sum(percent) == hp.logits_dim
        return percent.astype(int)

    def _get_seg_duration(self, seg_size: int) -> float:
        return seg_size * hp.hop_length / hp.sample_rate

    def _get_seg_tensor_from_list(self, audio_path_list: list[str]) -> torch.Tensor:
        with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as temp_file:
            params, nframes = self.audio_parser.merge(audio_path_list)
            self.audio_parser.write(temp_file, params, nframes)
            seg_tensors = self.audio_parser._get_seg_tensor_from_audio(temp_file.name)
        return seg_tensors

    def _get_seg_labels_from_list(self, audio_path_list: list[str]) -> torch.Tensor:
        _label = np.arange(len(audio_path_list))
        percent = self._get_percent(audio_path_list)
        label = np.repeat(_label, percent)
        return torch.tensor(label)


def collate_fn(batch):
    input_features, labels = [], []
    for row in batch:
        input_features.append(row[0])
        labels.append(row[1])
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
