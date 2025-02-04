import os
import random

import numpy as np

from src.utils import hp


class MixDataPipe:
    def __init__(self):
        self.max_speakers_in_audio = hp.max_speakers_in_audio
        self.min_speakers_in_audio = hp.min_speakers_in_audio

    def _get_file_list(self, _dir: str) -> list[str]:
        return [os.path.join(_dir, file_name) for file_name in os.listdir(_dir)]

    def remix(self, _dir: str) -> list[list[str]]:
        new_file_list = []
        file_list = self._get_file_list(_dir)
        while len(file_list) > 0:
            num_speakers = np.random.randint(
                self.min_speakers_in_audio, self.max_speakers_in_audio
            )

            num_speakers = min(num_speakers, len(file_list))
            idxs = random.sample(range(len(file_list)), k=num_speakers)

            sorted_idxs = sorted(idxs, reverse=True)
            files = [file_list.pop(idx) for idx in sorted_idxs]
            new_file_list.append(files)

        return new_file_list
