from typing import Union

import torch
from transformers import WhisperForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput

from src.utils import hp


class WhisperForDiarization(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = WhisperForConditionalGeneration.from_pretrained(hp.model_name)
        self.proj_out = torch.nn.Linear(hp.vocab_size, hp.proj_size, bias=False)

    def forward(
        self,
        input_features: torch.Tensor,
        labels: Union[torch.Tensor, None] = None,
    ) -> Seq2SeqLMOutput:

        outputs = self.base_model(input_features=input_features, labels=labels)

        lm_logits: torch.Tensor = self.proj_out(outputs.logits)

        loss = None
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()

            loss = loss_fn(lm_logits.view(-1, hp.proj_size), labels.view(-1))

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
