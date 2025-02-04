import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.utils import AudioParser, MixDataPipe, hp


class SpearkerData(Dataset):
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
        return mixed_file

    def _get_mixed_file_list(self, split: str) -> list[list[str]]:
        assert split in [
            "train",
            "test",
            "valid",
        ], f"Only support [train | test | valid]"

        mixed_file_list = self.mix_data_pipe.remix(hp.split2path[split])
        return mixed_file_list

    def _get_percent(self, input_path: list[str]) -> np.ndarray:
        percent = list()
        for audio_path in input_path:
            _, frames = self.audio_parser.read(audio_path)
            percent.append(len(frames))

        percent = (np.array(percent) / np.sum(percent) * hp.n_mels).round().astype(int)
        assert percent.sum() == hp.n_mels
        return percent

    def _get_seg_duration(self, seg_size: int) -> float:
        return seg_size * hp.hop_length / hp.sample_rate

    # copyed from https://github.com/openai/whisper/blob/517a43ecd132a2089d85f4ebc044728a71d49f6e/whisper/audio.py#L65
    def _get_truncate(self, mel_seg: torch.Tensor) -> torch.Tensor:
        mel_seg_size = mel_seg.size(-1)

        if mel_seg_size > hp.n_frames:
            mel_seg = mel_seg[:, : hp.n_frames]

        if mel_seg_size < hp.n_frames:
            pad_width = [0, hp.n_frames - mel_seg_size, 0, 0]
            mel_seg = F.pad(mel_seg, pad_width)
        return mel_seg

    def _get_seg_tensor_from_audio(self, audio_path: str) -> torch.Tensor:
        mel = self.audio_parser.get_log_mel(audio_path)

        content_frames = mel.size(-1) - hp.n_frames
        # content_duration = content_frames * hp.hop_length / hp.sample_rate

        seek_points = [0]

        if len(seek_points) % 2:
            seek_points.append(content_frames)
        seek_clips: list[tuple[int, int]] = list(
            zip(seek_points[::2], seek_points[1::2])
        )

        segs = []
        for start, end in seek_clips:
            # time_offset = self._get_seg_duration(start)
            # windows_end_time = (start + hp.n_frames) * hp.hop_length / hp.sample_rate
            seg_size = min(hp.n_frames, content_frames - start, end - start)

            # seg_duration = self._get_seg_duration(seg_size)
            mel_seg = self._get_truncate(mel[:, start : start + seg_size])
            segs.append(mel_seg)
        return torch.stack(segs)

    def _get_seg_tensor_from_list(self, audio_path_list: list[str]) -> torch.Tensor:
        seg_tensors = list()
        for audio_path in audio_path_list:
            seg_tensor = self._get_seg_tensor_from_audio(audio_path)
            seg_tensors.append(seg_tensor)
        return torch.concat(seg_tensors)
