from transformers import WhisperTokenizer


class WhisperTokenizerForDiarization(WhisperTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._add_speaker_token()

    def _add_speaker_token(self):
        self.add_tokens(["<speaker_changed>"])
