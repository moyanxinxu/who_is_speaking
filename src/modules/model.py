import os
from typing import Union

import torch
from transformers import WhisperModel
from transformers.modeling_outputs import ModelOutput

from src.utils import AudioParser, hp

from .tokenizer import WhisperTokenizerForDiarization


class WhisperForDiarization(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = WhisperTokenizerForDiarization()
        self.whisper = WhisperModel.from_pretrained(hp.model_name)
        self.proj_out = torch.nn.Linear(hp.decoder_out_size, hp.proj_size, bias=False)
        self.audio_parser = AudioParser()
        self._freeze_params()

    def _freeze_params(self):
        for param in self.whisper.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_features: torch.Tensor,
        labels: Union[torch.Tensor, None] = None,
        # attention_mask: Union[torch.Tensor, None] = None,
    ) -> ModelOutput:

        encoder_outputs = self.whisper.encoder(
            input_features=input_features,
        )

        if labels is not None:
            decoder_input_ids = self.tokenizer.shift(labels)

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

    def predict(self, audio_path: str):
        audio_features = self.audio_parser.get_seg_tensor_from_audio(audio_path)
        prefix_tokens = torch.Tensor([[self.tokenizer.bos_token]])

        encoder_outputs = self.whisper.encoder(input_features=audio_features)
        generate = [prefix_tokens.item()]

        for _ in range(hp.max_length):
            latent = self.whisper.decoder(
                input_ids=prefix_tokens,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
            )

            logits = self.proj_out(latent.last_hidden_state)
            generate.append(torch.argmax(logits, dim=-1).item())

            if generate[-1] == self.tokenizer.eos_token:
                break
        return generate

    def load_params(self, checkpoint: str):
        if os.path.isfile(checkpoint) and checkpoint.endswith(".pt"):
            params = torch.load(checkpoint)
            self.load_state_dict(params)
        else:
            raise FileNotFoundError("checkpoint file not valid")
