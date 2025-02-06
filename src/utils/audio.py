import wave
from subprocess import CalledProcessError, run

import numpy as np
import torch
import torch.nn.functional as F

from .config import hp


class AudioParser:
    def __init__(self):
        self.nchannels = hp.nchannels
        self.sampwidth = hp.sampwidth
        self.framerate = hp.framerate
        self.comptype = hp.comptype
        self.compname = hp.compname

    def read(self, in_path: str) -> tuple[wave._wave_params, bytes]:
        with wave.open(in_path, "rb") as file:
            params = file.getparams()
            frames = file.readframes(params.nframes)
            return params, frames

    def write(self, out_path: str, params: wave._wave_params, frames: bytes):
        with wave.open(out_path, "wb") as file:
            file.setparams(params)
            file.writeframes(frames)

    def merge(self, in_path_list: list[str]) -> tuple[wave._wave_params, bytes]:
        # ('nchannels', 'sampwidth', 'framerate', 'nframes', 'comptype', 'compname')
        nframes = 0
        frames = b""

        for in_path in in_path_list:
            _params, _frames = self.read(in_path)

            nframes += _params.nframes
            frames += _frames

        params = (
            self.nchannels,
            self.sampwidth,
            self.framerate,
            nframes,
            self.comptype,
            self.compname,
        )

        return wave._wave_params(*params), frames

    # copyed from https://github.com/openai/whisper/blob/517a43ecd132a2089d85f4ebc044728a71d49f6e/whisper/audio.py#L110
    def load_audio(self, file: str, sample_rate: int = hp.sample_rate):

        # ffmpeg -nostdin -threads 0 -i "path/to/audio/file" -f s16le -ac 1 -acodec pcm_s16le -ar 16000 -
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads",
            "0",
            "-i",
            file,
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate),
            "-",
        ]
        # fmt: on
        try:
            out = run(cmd, capture_output=True, check=True).stdout
        except CalledProcessError as e:
            raise RuntimeError(f"Failed to load audio") from e

        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    # copyed from https://github.com/openai/whisper/blob/517a43ecd132a2089d85f4ebc044728a71d49f6e/whisper/audio.py#L110
    def get_log_mel(
        self,
        audio_path: str,
        n_mels: int = hp.n_mels,
        padding: int = hp.n_samples,
    ) -> torch.Tensor:

        if isinstance(audio_path, str):
            audio = self.load_audio(audio_path)
            audio = torch.from_numpy(audio)

        if padding > 0:
            audio = F.pad(audio, (0, padding))
        window = torch.hann_window(hp.n_fft)
        stft = torch.stft(
            audio,
            hp.n_fft,
            hp.hop_length,
            window=window,
            return_complex=True,
        )
        magnitudes = stft[..., :-1].abs() ** 2

        filters = self.get_mel_filters(n_mels)
        mel_spec = filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec

    # copyed from https://github.com/openai/whisper/blob/517a43ecd132a2089d85f4ebc044728a71d49f6e/whisper/audio.py#L92
    def get_mel_filters(self, n_mels: int) -> torch.Tensor:
        assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

        filters_path = hp.filters_path
        with np.load(filters_path, allow_pickle=False) as f:
            return torch.from_numpy(f[f"mel_{n_mels}"])

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
        mel = self.get_log_mel(audio_path)

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
