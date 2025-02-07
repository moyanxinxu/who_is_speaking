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

    def get_audio_time(self, params: wave._wave_params) -> int:
        """四舍五入, 时间单位为秒"""
        return round(params.nframes / params.framerate)

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

    def get_audio_data_list(self, audio_list: list[str]) -> list[str]:
        audios = []
        for audio in audio_list:
            wav = self.load_audio(audio)
            audios.append(wav)
        return audios
