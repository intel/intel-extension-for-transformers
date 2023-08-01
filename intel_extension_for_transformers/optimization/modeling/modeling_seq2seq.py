#  Copyright (c) 2023 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import copy
import logging
from typing import Dict, Optional, Tuple, Union

import torch
from torch.nn import CrossEntropyLoss
from optimum.utils import NormalizedConfigManager
import transformers
from transformers import AutoConfig, AutoModelForSeq2SeqLM
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.utils import is_torch_fx_proxy
from optimum.intel.utils.import_utils import is_transformers_version
from .modeling_base_seq2seq import INCBaseModelForSeq2SeqLM


if is_transformers_version("<", "4.25.0"):
    from transformers.generation_utils import GenerationMixin
else:
    from transformers.generation import GenerationMixin


logger = logging.getLogger(__name__)

_TOKENIZER_FOR_DOC = "AutoTokenizer"

INPUTS_DOCSTRING = r"""
    Arguments:
        encoder (`JIT model or PyTorch model`):
            The PyTorch model associated to the encoder.
        decoder (`JIT model or PyTorch model`):
            The PyTorch model associated to the decoder.
        decoder_with_past (`JIT model or PyTorch model`):
            The PyTorch model associated  to the decoder with past key values.
        config (`transformers.PretrainedConfig`):
            [PretrainedConfig](https://huggingface.co/docs/transformers/main_classes/configuration#transformers.PretrainedConfig)
            is an instance of the configuration associated to the model. Initializing with a config file does
            not load the weights associated with the model, only the configuration.
"""

ENCODER_INPUTS_DOCSTRING = r"""
    Arguments:
        input_ids (`torch.LongTensor`):
            Indices of input sequence tokens in the vocabulary of shape `(batch_size, encoder_sequence_length)`.
        attention_mask (`torch.LongTensor`):
            Mask to avoid performing attention on padding token indices, of shape
            `(batch_size, encoder_sequence_length)`. Mask values selected in `[0, 1]`.
"""


DECODER_INPUTS_DOCSTRING = r"""
    Arguments:
        input_ids (`torch.LongTensor`):
            Indices of decoder input sequence tokens in the vocabulary of shape `(batch_size, decoder_sequence_length)`.
        encoder_hidden_states (`torch.FloatTensor`):
            The encoder `last_hidden_state` of shape `(batch_size, encoder_sequence_length, hidden_size)`.
        encoder_attention_mask (`torch.LongTensor`, *optional*):
            Mask to avoid performing cross-attention on padding tokens indices of encoder `input_ids`.
        past_key_values (`tuple(tuple(torch.FloatTensor), *optional*)`
            Contains the precomputed key and value hidden states of the attention blocks used to speed up decoding.
            The tuple is of length `config.n_layers` with each tuple having 2 tensors of shape
            `(batch_size, num_heads, decoder_sequence_length, embed_size_per_head)` and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
"""

SEQ2SEQ_MODEL_DOCSTRING = r"""
    Arguments:
        input_ids (`torch.LongTensor`):
            Indices of input sequence tokens in the vocabulary of shape `(batch_size, encoder_sequence_length)`.
        attention_mask (`torch.LongTensor`):
            Mask to avoid performing attention on padding token indices, of shape
            `(batch_size, encoder_sequence_length)`. Mask values selected in `[0, 1]`.
        decoder_input_ids (`torch.LongTensor`):
            Indices of decoder input sequence tokens in the vocabulary of shape `(batch_size, decoder_sequence_length)`.
        encoder_outputs (`torch.FloatTensor`):
            The encoder `last_hidden_state` of shape `(batch_size, encoder_sequence_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor), *optional*)`
            Contains the precomputed key and value hidden states of the attention blocks used to speed up decoding.
            The tuple is of length `config.n_layers` with each tuple having 2 tensors of shape
            `(batch_size, num_heads, decoder_sequence_length, embed_size_per_head)` and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
"""


TRANSLATION_EXAMPLE = r"""
    Example of text generation:
    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.intel import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> text = "He never went out without a book under his arm, and he often came back with two."
    >>> inputs = tokenizer(text, return_tensors="pt")
    >>> gen_tokens = model.generate(**inputs)
    >>> outputs = tokenizer.batch_decode(gen_tokens)
    ```

    Example using `transformers.pipeline`:
    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.intel import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> pipe = pipeline("translation_en_to_fr", model=model, tokenizer=tokenizer)
    >>> text = "He never went out without a book under his arm, and he often came back with two."
    >>> outputs = pipe(text)
    ```
"""


@add_start_docstrings(
    """
    Sequence-to-sequence model with a language modeling head for inference.
    """,
    INPUTS_DOCSTRING,
)
class INCModelForSeq2SeqLM(INCBaseModelForSeq2SeqLM, GenerationMixin):
    auto_model_class = AutoModelForSeq2SeqLM

    def __init__(
        self,
        encoder_model,
        decoder_model,
        decoder_with_past_model=None,
        config: transformers.PretrainedConfig = None,
        use_cache: bool = True,
        **kwargs,
    ):
        super().__init__(encoder_model=encoder_model, decoder_model=decoder_model,
                         decoder_with_past_model=decoder_with_past_model, config=config, **kwargs)
        self._device = torch.device("cpu")
        self.main_input_name = "input_ids"
        self.use_cache = use_cache
        self.normalized_config = NormalizedConfigManager.get_normalized_config_class(config.model_type)(config)
        self.model_dim = kwargs.pop("model_dim", None)
        self.model_parallel = kwargs.pop("model_parallel", None)
        self.encoder = INCEncoder(self.encoder_model, self._device, config)
        self.decoder = INCDecoder(self.decoder_model, self._device, config)
        self.decoder_with_past = INCDecoder(self.decoder_with_past_model, self._device, config)
        self.to(self._device)

        # Avoid warnings when creating a transformers pipeline
        AutoConfig.register(self.base_model_prefix, AutoConfig)
        self.auto_model_class.register(AutoConfig, self.__class__)

    @add_start_docstrings_to_model_forward(
        SEQ2SEQ_MODEL_DOCSTRING.format("batch_size, sequence_length")
        + TRANSLATION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="INCModelForSeq2SeqLM",
            checkpoint="fabiochiu/t5-base-tag-generation",
        )
    )
    def forward(
        self,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        # Encode if needed : first prediction pass
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        labels = kwargs.get("labels", None)
        decoder_inputs_embeds = kwargs.get("decoder_inputs_embeds", None)
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Decode
        encoder_hidden_states = encoder_outputs[0] \
            if isinstance(encoder_outputs, tuple) else encoder_outputs.last_hidden_state
        if past_key_values is None:
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
            )
            if not self.config.return_dict:
                if isinstance(decoder_outputs, tuple):
                    return decoder_outputs
                else:
                    return decoder_outputs.values()
            if isinstance(decoder_outputs, tuple):
                return Seq2SeqLMOutput(logits=decoder_outputs[0], past_key_values=decoder_outputs[1])
            return decoder_outputs
        elif self.decoder_with_past.model is None:
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                past_key_values=past_key_values,
                encoder_hidden_states=encoder_hidden_states,
            )
            if not self.config.return_dict:
                if isinstance(decoder_outputs, tuple):
                    return decoder_outputs
                else:
                    return decoder_outputs.values()
            if isinstance(decoder_outputs, tuple):
                return Seq2SeqLMOutput(logits=decoder_outputs[0], past_key_values=decoder_outputs[1])
            return decoder_outputs
        else:
            decoder_outputs = self.decoder_with_past(
                input_ids=decoder_input_ids,
                past_key_values=past_key_values,
                encoder_hidden_states=encoder_hidden_states,
            )
            if not self.config.return_dict:
                if isinstance(decoder_outputs, tuple):
                    return decoder_outputs
                else:
                    return decoder_outputs.values()
            if isinstance(decoder_outputs, tuple):
                return Seq2SeqLMOutput(logits=decoder_outputs[0], past_key_values=decoder_outputs[1])
            return decoder_outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ) -> Dict:
        past_key_values = past_key_values or kwargs.get("past", None)

        if self.use_cache and past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values or kwargs.get("past", None),
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def get_encoder(self):
        return self.encoder

    # Copied from transformers.models.bart.modeling_bart.BartForConditionalGeneration._reorder_cache
    @staticmethod
    def _reorder_cache(past, beam_idx) -> Tuple[Tuple[torch.FloatTensor]]:
        reordered_past = ()
        for layer_past in past:
            # Cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. "
                "In T5 it is usually set to the pad_token_id. See T5 docs for more information."
            )

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    @property
    def encoder_model(self):
        return self._encoder_model

    @encoder_model.setter
    def encoder_model(self, model):
        self._encoder_model = model
        self.encoder.model = model

    @property
    def decoder_model(self):
        return self._decoder_model

    @decoder_model.setter
    def decoder_model(self, model):
        self._decoder_model = model
        self.decoder.model = model

    @property
    def decoder_with_past_model(self):
        return self._decoder_with_past_model

    @decoder_with_past_model.setter
    def decoder_with_past_model(self, model):
        self._decoder_with_past_model = model
        self.decoder_with_past.model = model


class INCEncoder(torch.nn.Module):
    """
    Encoder model for inference.

    Arguments:
        model (PyTorch model or neural-compressor model):
            The model to inference.
        device (`torch.device`):
            The device type used by this process.
    """

    def __init__(self, model, device: str, config):
        super(INCEncoder, self).__init__()
        self.model = model
        self._device = device
        self.device = torch.device("cpu")
        self.main_input_name = "input_ids"
        self.config = copy.deepcopy(config)

    @add_start_docstrings_to_model_forward(ENCODER_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor = None,
        **kwargs,
    ) -> BaseModelOutput:

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        outputs = self.model(**inputs)

        if not self.config.return_dict:
            if not isinstance(outputs, tuple):
                outputs = tuple([outputs.last_hidden_state])
            return outputs

        if isinstance(outputs, tuple):
            last_hidden_state = outputs[0].to(self.device)
            return BaseModelOutput(last_hidden_state=last_hidden_state)
        return outputs

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def to(self, device: Union[torch.device, str]):
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        self.model.to(self._device)
        return self


class INCDecoder(torch.nn.Module):
    """
    Decoder model for inference.

    Arguments:
        model (PyTorch model or neural-compressor model):
            The model to inference.
        device (`torch.device`):
            The device type used by this process.
        config (PretrainedConfig):
            The config of model.
    """

    def __init__(self, model, device: str, config):
        super(INCDecoder, self).__init__()
        self.model = model
        self._device = device
        self.device = torch.device("cpu")
        self.config = copy.deepcopy(config)

    @add_start_docstrings_to_model_forward(DECODER_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor,
        encoder_hidden_states: torch.FloatTensor,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        return_dict = True,
        **kwargs
    ) -> Seq2SeqLMOutput:
        inputs = {}

        inputs["decoder_input_ids"] = input_ids

        # Add the encoder_hidden_states inputs when needed
        if encoder_hidden_states is not None:
            encoder_outputs = []
            encoder_outputs.append(encoder_hidden_states)
            inputs["encoder_outputs"] = tuple(encoder_outputs)

        if past_key_values is not None:
            inputs["past_key_values"] = past_key_values

        # Run inference
        outputs = self.model(**inputs)

        if not self.config.return_dict:
            if isinstance(outputs, Seq2SeqLMOutput):
                outputs = (outputs.logits, outputs.past_key_values)
            return outputs

        if isinstance(outputs, Seq2SeqLMOutput):
            return outputs

        return Seq2SeqLMOutput(logits=outputs[0], past_key_values=outputs[1])

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def to(self, device: Union[torch.device, str]):
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        self.model.to(self._device)
        return self
