import wave


class WavParser:
    def __init__(self):
        pass

    def read(self, in_path: str):
        with wave.open(in_path, "rb") as file:
            params = file.getparams()
            frames = file.readframes(params.nframes)
            return params, frames

    def write(self, out_path: str, params: wave._wave_params, frames: bytes):
        with wave.open(out_path, "wb") as file:
            file.setparams(params)
            file.writeframes(frames)

    def merge(self, in_path_list: list[str]):
        # ('nchannels', 'sampwidth', 'framerate', 'nframes', 'comptype', 'compname')
        nframes = 0
        frames = b""

        for in_path in in_path_list:
            _params, _frames = self.read(in_path)

            nframes += _params.nframes
            frames += _frames

        params = (1, 2, 16000, nframes, "NONE", "not compressed")

        return wave._wave_params(*params), frames
