import os
from typing import Union

import torch
from transformers import WhisperModel, WhisperProcessor
from transformers.modeling_outputs import ModelOutput

from src.utils import AudioParser, hp

from .tokenizer import WhisperTokenizerForDiarization


class WhisperForDiarization(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = WhisperTokenizerForDiarization()
        self.whisper = WhisperModel.from_pretrained(hp.model_name)
        self.audio_parser = AudioParser()
        self.proj_out = torch.nn.Linear(hp.decoder_out_size, hp.proj_size, bias=False)
        self.processor = WhisperProcessor.from_pretrained(hp.model_name)
        self._freeze_params()

    def _freeze_params(self):
        self.whisper.encoder._freeze_parameters()

    def forward(
        self,
        input_features: torch.Tensor,
        labels: Union[torch.Tensor, None] = None,
    ) -> ModelOutput:

        encoder_outputs = self.whisper.encoder(
            input_features=input_features,
        )

        decoder_input_ids = self.tokenizer.shift(labels) if labels is not None else None

        decoder_outputs = self.whisper.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
        )

        logits: torch.Tensor = self.proj_out(decoder_outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()

            loss = loss_fn(logits.view(-1, hp.proj_size), labels.view(-1))

        return ModelOutput(
            loss=loss,
            logits=logits,
            # past_key_values=outputs.past_key_values,
            # decoder_hidden_states=outputs.decoder_hidden_states,
            # decoder_attentions=outputs.decoder_attentions,
            # cross_attentions=outputs.cross_attentions,
            # encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            # encoder_hidden_states=outputs.encoder_hidden_states,
            # encoder_attentions=encoder_outputs.encoder_attentions,
        )

    @torch.inference_mode()
    def predict(self, audio_path: str):
        input_ids = torch.tensor([[self.tokenizer.bos_token]], dtype=torch.long)

        input_features = self.processor(
            self.audio_parser.load_audio(audio_path),
            sampling_rate=hp.sample_rate,
            return_tensors="pt",
        ).input_features

        encoder_outputs = self.whisper.encoder(input_features=input_features)

        for _ in range(hp.max_target_length):
            decoder_outputs = self.whisper.decoder(
                input_ids=input_ids,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
            )

            last_logits = self.proj_out(decoder_outputs.last_hidden_state[:, -1, :])

            next_token = torch.argmax(last_logits, dim=-1, keepdim=True)

            if next_token.item() == self.tokenizer.eos_token:
                break

            input_ids = torch.cat([input_ids, next_token], dim=-1)

        return self.tokenizer.decode(input_ids[0].tolist())

    def load_params(self, checkpoint: str):
        if os.path.isfile(checkpoint) and checkpoint.endswith(".pt"):
            params = torch.load(checkpoint)
            self.load_state_dict(params)
            print(f"load checkpoint from {checkpoint}")
        else:
            raise FileNotFoundError("checkpoint file not valid")
