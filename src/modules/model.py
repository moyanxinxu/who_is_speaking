import os
from typing import Union

import torch
from transformers import WhisperForConditionalGeneration
from transformers.modeling_outputs import ModelOutput

from src.utils import AudioParser, hp


class WhisperForDiarization(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = WhisperForConditionalGeneration.from_pretrained(hp.model_name)
        self.proj_out = torch.nn.Linear(hp.vocab_size, hp.proj_size, bias=False)
        self.audio_parser = AudioParser()
        self._freeze_params()  # A custom method to freeze the model parameters

    def forward(
        self,
        input_features: torch.Tensor,
        labels: Union[torch.Tensor, None] = None,
        decoder_input_ids: Union[torch.Tensor, None] = None,
        return_output_ids: bool = False,
    ) -> ModelOutput:

        outputs = self.base_model(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )

        lm_logits: torch.Tensor = self.proj_out(outputs.logits)

        loss = None
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()

            loss = loss_fn(lm_logits.view(-1, hp.proj_size), labels.view(-1))

        output_ids = None
        if return_output_ids:
            output_ids = torch.argmax(lm_logits, dim=-1)

        return ModelOutput(
            loss=loss,
            logits=lm_logits,
            output_ids=output_ids,
            # past_key_values=outputs.past_key_values,
            # decoder_hidden_states=outputs.decoder_hidden_states,
            # decoder_attentions=outputs.decoder_attentions,
            # cross_attentions=outputs.cross_attentions,
            # encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            # encoder_hidden_states=outputs.encoder_hidden_states,
            # encoder_attentions=outputs.encoder_attentions,
        )

    def _freeze_params(self):
        self.base_model.requires_grad_(False)
        self.proj_out.requires_grad_(True)

    def load_params(self, checkpoint: str):
        if os.path.isfile(checkpoint) and checkpoint.endswith(".pt"):
            params = torch.load(checkpoint)
            self.load_state_dict(params)
        else:
            raise FileNotFoundError("checkpoint file not valid")
