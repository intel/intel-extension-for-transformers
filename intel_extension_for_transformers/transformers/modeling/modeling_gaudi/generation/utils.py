# coding=utf-8
# Copyright 2022 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import inspect
import math
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from transformers.generation.beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers.generation.candidate_generator import CandidateGenerator
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation.utils import (
    NEED_SETUP_CACHE_CLASSES_MAPPING,
    GenerateBeamDecoderOnlyOutput,
    GenerateBeamEncoderDecoderOutput,
    GenerateBeamOutput,
    GenerateDecoderOnlyOutput,
    GenerateEncoderDecoderOutput,
    GenerateNonBeamOutput,
    GenerateOutput,
    GenerationMixin,
    GenerationMode,
    _split_model_inputs,
    stack_model_outputs,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import ModelOutput

from optimum.utils import logging

from optimum.habana.utils import HabanaProfile
from optimum.habana.transformers.integrations.deepspeed import unwrap_deepspeed_model
from .configuration_utils import GaudiGenerationConfig


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from .streamers import BaseStreamer


MODELS_OPTIMIZED_WITH_STATIC_SHAPES = [
    "bloom",
    "gpt2",
    "opt",
    "gptj",
    "gpt_neox",
    "llama",
    "falcon",
    "codegen",
    "gpt_bigcode",
    "bart",
    "mpt",
    "t5",
    "mistral",
    "phi",
    "mixtral",
    "blip_text_model",
]


logger = logging.get_logger(__name__)


def incrementor(bucket_size, prompt_len):
    assert bucket_size > 0
    passnum = -1
    while True:
        passnum += 1
        if passnum == 0:
            token_idx = prompt_len
            allocated_space = int(math.ceil(prompt_len / bucket_size) * bucket_size)
            if prompt_len % bucket_size == 0:
                allocated_space += bucket_size
            need_expansion = True
        else:
            token_idx += 1
            need_expansion = token_idx >= allocated_space
            if need_expansion:
                assert (allocated_space - token_idx) <= bucket_size
                allocated_space += bucket_size
        yield {
            "allocated_space": allocated_space,
            "passnum": passnum,
            "token_idx": token_idx,
            "need_expansion": need_expansion,
        }


class GaudiGenerationMixin(GenerationMixin):
    """This class enables to perform fast generation in lazy mode and with HPU graphs.

    The only difference with GenerationMixin is that the various generation
    methods will generate sequences whose size is max_length. Having constant
    sizes allows to make the most of lazy mode and HPU graphs.
    """

    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...].

        The tensor `token_idx` is not expanded.
        """

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    dict_to_expand[key] is not None
                    and key != "token_idx"
                    and key != "decoder_input_ids"
                    and isinstance(dict_to_expand[key], torch.Tensor)
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs

    def _get_hpu_graphs_kwargs(self, model_kwargs):
        hpu_graphs_kwargs = {}
        if model_kwargs["limit_hpu_graphs"]:
            hpu_graphs_kwargs.update({"bypass_hpu_graphs": False})
            if "first_token" not in model_kwargs.keys():
                model_kwargs["first_token"] = True
                hpu_graphs_kwargs.update({"bypass_hpu_graphs": True})
        return hpu_graphs_kwargs

    def _prepare_decoder_attention_mask(
        self,
        max_steps: int,  # current stopping criteria
        batch_size: int,
        pad_token_id: int,
        device: str,
        dtype: str = bool,
    ) -> torch.Tensor:
        x = torch.zeros((batch_size, max_steps), device=device, dtype=dtype)
        return x.index_fill(1, torch.tensor([0]), 1)  # First the position with pad_token_id

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        model_input_name: str,
        model_kwargs: Dict[str, torch.Tensor],
        decoder_start_token_id: Union[int, List[int]] = None,
        bos_token_id: int = None,
        device: torch.device = None,
        max_new_tokens: int = None,
    ) -> Tuple[torch.LongTensor, Dict[str, torch.Tensor]]:
        """Prepares `decoder_input_ids` for generation with encoder-decoder models."""
        # 1. Check whether the user has defined `decoder_input_ids` manually. To facilitate in terms of input naming,
        # we also allow the user to pass it under `input_ids`, if the encoder does not use it as the main input.

        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
        elif "input_ids" in model_kwargs and model_input_name != "input_ids":
            decoder_input_ids = model_kwargs.pop("input_ids")
        else:
            decoder_input_ids = None

        token_idx = model_kwargs.get("token_idx", None)

        # 2. Encoder-decoder models expect the `decoder_input_ids` to start with a special token. Let's ensure that.
        decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
        if device is None:
            device = self.device
        if token_idx is None:
            if isinstance(decoder_start_token_id, list):
                if len(decoder_start_token_id) != batch_size:
                    raise ValueError(f"`decoder_start_token_id` expected to have \
                      length {batch_size} but got {len(decoder_start_token_id)}"
                    )
                decoder_input_ids_start = torch.tensor(decoder_start_token_id, dtype=torch.long, device=device)
                decoder_input_ids_start = decoder_input_ids_start.view(-1, 1)
            else:
                decoder_input_ids_start = (
                    torch.ones((batch_size, 1), dtype=torch.long, device=device) * decoder_start_token_id
                )
        else:
            # creating padded decoder_input_ids to achieve static shapes.
            # Later new tokens once generated are copied in to decoder_input_ids based on token_idx
            max_length = max_new_tokens + 1 if max_new_tokens is not None else self.generation_config.max_length
            decoder_input_ids_start = (
                torch.ones((batch_size, max_length), dtype=torch.long, device=device) * decoder_start_token_id
            )

        # no user input -> use decoder_start_token_id as decoder_input_ids
        if decoder_input_ids is None:
            decoder_input_ids = decoder_input_ids_start
        # exception: Donut checkpoints have task-specific decoder starts and don't expect a BOS token
        elif self.config.model_type == "vision-encoder-decoder" and "donut" in self.name_or_path.lower():
            pass
        elif self.config.model_type in ["whisper"]:
            pass
        # user input but doesn't start with decoder_start_token_id -> prepend decoder_start_token_id (and adjust
        # decoder_attention_mask if provided)
        elif (
            isinstance(decoder_start_token_id, int)
            and (decoder_input_ids[:, 0] != decoder_start_token_id).all().item()
        ) or (
            isinstance(decoder_start_token_id, torch.Tensor)
            and (decoder_input_ids[:, 0] != decoder_start_token_id[:, 0]).all().item()
        ):
            decoder_input_ids = torch.cat([decoder_input_ids_start, decoder_input_ids], dim=-1)
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                decoder_attention_mask = torch.cat(
                    (torch.ones_like(decoder_attention_mask)[:, :1], decoder_attention_mask),
                    dim=-1,
                )
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask
        return decoder_input_ids, model_kwargs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        """Adds support for `token_idx`, which is necessary for using static shapes."""
        # mark to identify starting from second token
        model_kwargs["first_token"] = False
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        token_idx = model_kwargs.get("token_idx", None)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                if token_idx is not None:
                    attention_mask.index_fill_(1, token_idx, 1)
                else:
                    attention_mask = torch.cat(
                        [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                    )
                model_kwargs["attention_mask"] = attention_mask
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                if token_idx is not None:
                    decoder_attention_mask.index_fill_(1, token_idx, 1)
                else:
                    decoder_attention_mask = torch.cat(
                        [
                            decoder_attention_mask,
                            decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1)),
                        ],
                        dim=-1,
                    )
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask

        if token_idx is not None:
            token_idx.add_(1)
            if "token_idx_cpu" in model_kwargs:
                model_kwargs["token_idx_cpu"] += 1

        return model_kwargs

    @torch.no_grad()
    def update_model_kwargs_for_bucketing(
        self, params, input_ids, model_kwargs, pad_token_id, bucket_size, reduce_recompile=False
    ):
        if params["need_expansion"]:
            # Pad inputs to have static shapes during generation,
            # this gives better performance than dynamic shapes on HPUs
            pad_amount = params["allocated_space"] - input_ids.shape[-1]
            input_ids = torch.nn.functional.pad(input_ids, (0, pad_amount), value=pad_token_id)
            if model_kwargs["attention_mask"] is not None:
                model_kwargs["attention_mask"] = torch.nn.functional.pad(
                    model_kwargs["attention_mask"], (0, pad_amount), value=0
                )
            else:
                assert False, "Not tested for cases where attn_mask isn't passed"
            if reduce_recompile and params["passnum"] == 0:
                position_ids_cpu = model_kwargs["attention_mask"].long().cumsum(-1) - 1
                position_ids_cpu.masked_fill_(model_kwargs["attention_mask"] == 0, 1)
                input_ids = input_ids.to(self.device)
                model_kwargs["attention_mask"] = model_kwargs["attention_mask"].to(self.device)

            if "past_key_values" in model_kwargs:

                def create_pad_arg(pad_amount, i, j):
                    if model_kwargs["past_key_values"][0][0].dim() == 3:
                        assert self.config.model_type == "bloom"
                        if j == 0:
                            return (0, pad_amount)
                        elif j == 1:
                            return (0, 0, 0, pad_amount)
                        else:
                            assert False
                    elif model_kwargs["past_key_values"][0][0].dim() == 4:
                        return (0, 0, 0, pad_amount)  # llama, falcon
                    else:
                        assert False, "Unknown case, please handle, or dont use bucketing"

                new_kv = [None for i in range(len(model_kwargs["past_key_values"]))]
                if self.config.model_type == "gpt_bigcode" and model_kwargs["past_key_values"][0][0].dim() == 2:
                    # GPT_BIGCODE's kv cache is list of tensors.
                    new_kv = [None for i in range(len(model_kwargs["past_key_values"]))]
                    for i in range(len(model_kwargs["past_key_values"])):
                        pad = (0, 0, 0, pad_amount)
                        new_kv[i] = torch.nn.functional.pad(
                            model_kwargs["past_key_values"][i], pad, value=pad_token_id
                        )
                    model_kwargs["past_key_values"] = list(new_kv)
                else:
                    for i in range(len(model_kwargs["past_key_values"])):
                        tmp_lst = [None for j in range(len(model_kwargs["past_key_values"][i]))]
                        for j in range(len(model_kwargs["past_key_values"][i])):
                            pad_tuple = create_pad_arg(pad_amount, i, j)
                            # Different models might have different shapes of kv-cache
                            # create_pad_arg handles them on a per-model basis
                            # This is a necessary (but not sufficient) condition:
                            # what ever dimension we are padding, should be a multiple of bucket_size
                            # This check is added in case we get a new model with a new kv-cache structure,
                            # and we attempt to pad some wrong dimension
                            assert (
                                model_kwargs["past_key_values"][i][j].shape[-(len(pad_tuple) // 2)] % bucket_size == 0
                            )
                            tmp_lst[j] = torch.nn.functional.pad(
                                model_kwargs["past_key_values"][i][j], pad_tuple, value=pad_token_id
                            )
                        new_kv[i] = tuple(tmp_lst)
                    model_kwargs["past_key_values"] = tuple(new_kv)

        if "token_idx" not in model_kwargs:
            model_kwargs["token_idx"] = torch.tensor(params["token_idx"], device=self.device)
        return input_ids, model_kwargs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GaudiGenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        lazy_mode: Optional[bool] = False,
        hpu_graphs: Optional[bool] = False,
        profiling_warmup_steps: Optional[int] = 0,
        profiling_steps: Optional[int] = 0,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        r"""Generates sequences of token ids for models with a language modeling head.

        <Tip warning={true}>

        Most generation-controlling parameters are set in [`transformers.generation.generation_config`]
        which, if not passed, will be set to the model's default generation configuration.
        You can override any `generation_config` by passing the corresponding
        parameters to generate, e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](../generation_strategies).

        </Tip>

        Most of these parameters are explained in more detail in [this blog
        post](https://huggingface.co/blog/how-to-generate).
        Parameters:
            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config (`transformers.generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. If your stopping criteria depends on the `scores` input, make
                sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is
                intended for advanced users.
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            synced_gpus (`bool`, *optional*):
                Whether to continue running the while loop until max_length. Unless overridden this flag will be set to
                `True` under DeepSpeed ZeRO Stage 3 multiple GPUs environment to avoid hanging if one GPU finished
                generating before other GPUs. Otherwise it'll be set to `False`.
            assistant_model (`PreTrainedModel`, *optional*):
                An assistant model that can be used to accelerate generation. The assistant model must have the exact
                same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistant model
                is much faster than running generation with the model you're calling generate from. As such, the
                assistant model should be much smaller.
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            negative_prompt_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                The negative prompt needed for some processors such as CFG. The batch size must match the input batch
                size. This is an experimental feature, subject to breaking API changes in future versions.
            negative_prompt_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Attention_mask for `negative_prompt_ids`.
            lazy_mode (`bool`, *optional*, defaults to `False`):
                Whether the run is executed in lazy mode or not (i.e. eager mode).
            hpu_graphs (`bool`, *optional*, defaults to `False`):
                Whether to use HPU graphs for inference.
            profiling_warmup_steps (`int`, *optional*, defaults to 0):
                Number of steps to ignore for profiling.
            profiling_steps (`int`, *optional*, defaults to 0):
                Number of steps to be captured when enabling profiling.
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`transformers.utils.ModelOutput`] or `torch.LongTensor`:
            A [`transformers.generationutils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.
                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`),
            the possible [`transformers.generationutils.ModelOutput`] types are:
                    - [`transformers.generation.GenerateDecoderOnlyOutput`],
                    - [`transformers.generation.GenerateBeamDecoderOnlyOutput`]
                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`),
                the possible [`transformers.generationutils.ModelOutput`] types are:
                    - [`transformers.generation.GenerateEncoderDecoderOutput`],
                    - [`transformers.generation.GenerateBeamEncoderDecoderOutput`]
        """
        if synced_gpus is None:
            if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
                synced_gpus = True
            else:
                synced_gpus = False

        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()
        if hpu_graphs and not lazy_mode:
            raise ValueError(
                "`hpu_graphs` is True but `lazy_mode` is False. HPU graphs require `lazy_mode` to be set to True."
            )

        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        if generation_config is None:
            # legacy: users may modify the model configuration to control generation. To trigger this legacy behavior,
            # three conditions must be met
            # 1) the generation config must have been created from the model config (`_from_model_config` field);
            # 2) the generation config must have seen no modification since its creation (the hash is the same);
            # 3) the user must have set generation parameters in the model config.
            if (
                self.generation_config._from_model_config
                and self.generation_config._original_object_hash == hash(self.generation_config)
                and self.config._has_non_default_generation_parameters()
            ):
                new_generation_config = GaudiGenerationConfig.from_model_config(self.config)
                if new_generation_config != self.generation_config:
                    warnings.warn(
                        "You have modified the pretrained model configuration to control generation."
                        "This is a deprecated strategy to control generation and will be removed in a future version."
                    )
                    self.generation_config = new_generation_config
            generation_config = self.generation_config

        generation_config = copy.deepcopy(generation_config)
        if generation_config.static_shapes is None:
            generation_config.static_shapes = self.config.model_type in MODELS_OPTIMIZED_WITH_STATIC_SHAPES
        self.generation_config.static_shapes = generation_config.static_shapes
        if generation_config.ignore_eos is None:
            generation_config.ignore_eos = kwargs.get("ignore_eos", lazy_mode)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
        if self.config.model_type == "falcon" and "token_type_ids" in kwargs.keys():
            for key in ["token_type_ids"]:
                model_kwargs.pop(key, None)
        self._validate_model_kwargs(model_kwargs.copy())

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get("attention_mask", None) is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            logger.warning(
                f"Setting `pad_token_id` to `eos_token_id`:{generation_config.eos_token_id} for open-end generation."
            )
            generation_config.pad_token_id = eos_token_id

        # 3. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        # 4. Define other model kwargs
        model_kwargs["output_attentions"] = generation_config.output_attentions
        model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            model_kwargs["use_cache"] = True
        else:
            model_kwargs["use_cache"] = generation_config.use_cache
        self.generation_config.max_length = generation_config.max_length
        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
            )

        is_greedy_or_beam_and_bucket = (
            not generation_config.bucket_internal
            and generation_config.bucket_size > 0
            and (
                self._get_generation_mode(generation_config, assistant_model) == GenerationMode.GREEDY_SEARCH
                or self._get_generation_mode(generation_config, assistant_model) == GenerationMode.BEAM_SEARCH
            )
        )
        model_kwargs["bucket_size"] = generation_config.bucket_size if generation_config.static_shapes else -1
        model_kwargs["bucket_internal"] = generation_config.bucket_internal
        model_kwargs["reduce_recompile"] = (
            generation_config.reduce_recompile if generation_config.reduce_recompile is not None else False
        )
        if model_kwargs["reduce_recompile"]:
            assert generation_config.bucket_size
        if generation_config.reuse_cache:
            assert self.config.model_type in [
                "llama",
                "mistral",
                "falcon",
            ], "reuse_cache only supported by llama, mistral and falcon at the moment"
            if not generation_config.bucket_internal:
                assert (
                    generation_config.bucket_size <= 0
                ), "please set bucket_internal along with reuse_cache and bucket_size"
            else:
                assert generation_config.bucket_size >= 0, "please set valid bucket_size to use bucket_internal"

        if generation_config.static_shapes:
            # Pad inputs to have static shapes during generation,
            # this gives better performance than dynamic shapes on HPUs
            # In encoder_decoder models, Inputs are already padded

            if not self.config.is_encoder_decoder:
                # only pad if bucket_size < -1. If we are bucketing (bucket_size > 0),
                # then that is taken care in greedy_search()
                if not is_greedy_or_beam_and_bucket:
                    # token_idx is the current index in the generation process,
                    # it is incremented each time a new token is generated
                    token_idx = inputs_tensor.shape[-1]
                    model_kwargs["token_idx"] = torch.tensor(token_idx, device=inputs_tensor.device)
                    model_kwargs["token_idx_cpu"] = token_idx
                    if generation_config.max_new_tokens is None:
                        generation_config.max_new_tokens = generation_config.max_length - token_idx
                    inputs_tensor = torch.nn.functional.pad(
                        inputs_tensor, (0, generation_config.max_new_tokens), value=generation_config.pad_token_id
                    )
                    for other_inputs in ["attention_mask", "token_type_ids"]:
                        if model_kwargs.get(other_inputs) is not None:
                            model_kwargs[other_inputs] = torch.nn.functional.pad(
                                model_kwargs[other_inputs], (0, generation_config.max_new_tokens), value=0
                            )
            else:
                assert generation_config.bucket_size <= 0, "Untested path for bucket>0"
                token_idx = 1
                model_kwargs["token_idx"] = torch.tensor(token_idx, device=inputs_tensor.device)
                if model_kwargs.get("decoder_attention_mask", None) is None and generation_config.use_cache:
                    max_length = (
                        generation_config.max_new_tokens + 1
                        if generation_config.max_new_tokens is not None
                        else generation_config.max_length
                    )
                    model_kwargs["decoder_attention_mask"] = self._prepare_decoder_attention_mask(
                        max_length,
                        inputs_tensor.shape[0],
                        generation_config.pad_token_id,
                        inputs_tensor.device,
                    )

        # decoder-only models should use left-padding for generation
        if not self.config.is_encoder_decoder:
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if generation_config.pad_token_id is not None:
                position = model_kwargs["token_idx"] - 1 if "token_idx" in model_kwargs else -1
                if (
                    len(inputs_tensor.shape) == 2
                    and torch.sum(inputs_tensor[:, position] == generation_config.pad_token_id) > 0
                ):
                    logger.warning(
                        "A decoder-only architecture is being used, but right-padding was detected! For correct "
                        "generation results, please set `padding_side='left'` when initializing the tokenizer."
                    )

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config.decoder_start_token_id,
                bos_token_id=generation_config.bos_token_id,
                device=inputs_tensor.device,
                max_new_tokens=generation_config.max_new_tokens,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            if "token_idx" in model_kwargs:
                generation_config.max_length = input_ids_length
            else:
                generation_config.max_length = generation_config.max_new_tokens + input_ids_length
        # otherwise the total length [inputs-embeds-len + new-tokens-len] will go beyond indicated `max_length`
        elif (
            model_input_name == "inputs_embeds"
            and inputs_tensor.shape[:-1] != input_ids.shape
            and not self.config.is_encoder_decoder
        ):
            generation_config.max_length -= inputs_tensor.shape[1]

        # if we don't pass `past_key_values` and a cache_implementation is specified
        if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING and not model_kwargs.get(
            "past_key_values", False
        ):
            cache_cls = NEED_SETUP_CACHE_CLASSES_MAPPING[generation_config.cache_implementation]
            if not callable(getattr(self, "_setup_cache", None)):
                raise ValueError(
                    "The `generation_config` defines a `cache_implementation` that is not compatible with this model."
                    " Make sure it has a `_setup_cache` function."
                )
            self._setup_cache(cache_cls, max_batch_size=batch_size, max_cache_len=generation_config.max_length)

        self._validate_generated_length(
            generation_config,
            model_kwargs["token_idx"].item() if "token_idx" in model_kwargs else input_ids_length,
            has_default_max_length,
        )

        # determine whether introduce trim_logits feature
        model_kwargs["trim_logits"] = generation_config.trim_logits

        # determine whether attention softmax needs to execute in lower precision
        model_kwargs["attn_softmax_bf16"] = generation_config.attn_softmax_bf16

        # determine whether limit_hpu_graphs needs to be used
        model_kwargs["limit_hpu_graphs"] = generation_config.limit_hpu_graphs

        # prepare for allocate kv cache
        model_kwargs["reuse_cache"] = generation_config.reuse_cache

        # determine whether flash attention needs to be used
        model_kwargs["use_flash_attention"] = generation_config.use_flash_attention
        model_kwargs["flash_attention_recompute"] = True if generation_config.flash_attention_recompute else False
        model_kwargs["flash_attention_causal_mask"] = True if generation_config.flash_attention_causal_mask else False

        if not self.config.is_encoder_decoder:
            calculated_max_length = input_ids.shape[-1]
            if not generation_config.static_shapes and generation_config.max_new_tokens is not None:
                calculated_max_length = input_ids.shape[-1] + generation_config.max_new_tokens
            if generation_config.use_cache and generation_config.reuse_cache:
                bs, _ = input_ids.shape
                # attention_sinks has fixed kv_cache_len
                if generation_config.attention_sink_size is not None and \
                    generation_config.attention_sink_window_size is not None:
                    attn_window_len = generation_config.attention_sink_size + \
                            generation_config.attention_sink_window_size
                    calculated_max_length = max(calculated_max_length, attn_window_len)
                if not is_greedy_or_beam_and_bucket:
                    unwrap_deepspeed_model(self).allocate_kv_cache(
                        bs * generation_config.num_beams, calculated_max_length, token_idx
                    )
                    model_kwargs["kv_cache_len"] = calculated_max_length

            if self.config.model_type in ["llama", "falcon"]:
                if self.config.max_position_embeddings < calculated_max_length:
                    unwrap_deepspeed_model(self).update_sincos_cache(seq_len=calculated_max_length)

        # 7. determine generation mode
        generation_mode = self._get_generation_mode(generation_config, assistant_model)
        if generation_config.bucket_size > 0:
            assert generation_config.static_shapes, "bucket_size > 0 can be set only when static_shapes is set"
        # if generation_config.bucket_size <= 0, padding is handled by the generating fn (like greedy_search)
        if generation_config.static_shapes and generation_config.bucket_size > 0:
            assert (
                generation_mode == GenerationMode.GREEDY_SEARCH or generation_mode == GenerationMode.BEAM_SEARCH
            ), "generation_config.bucket_size > 0 supported only for greedy mode"

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if self.device.type != input_ids.device.type:
            warnings.warn(
                (
                    "You are calling .generate() with the `input_ids` being on a device type different"
                    f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                    f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                    " Please make sure that you have put `input_ids` to the"
                    f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                    " running `.generate()`."
                ),
                UserWarning,
            )

        # 8. prepare distribution pre_processing samplers
        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        # 9. prepare stopping criteria
        self.generation_config.generation_mode = generation_mode
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )

        # In lazy mode, import Habana torch to be able to add mark_step()
        if lazy_mode:
            import habana_frameworks.torch.core as htcore

            self.htcore_generation = htcore

        # 10. go into different generation modes
        if generation_mode == GenerationMode.ASSISTED_GENERATION:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing assisted generate, "
                    f"but is {generation_config.num_return_sequences}."
                )
            if batch_size > 1:
                raise ValueError("assisted generate is only supported for batch_size = 1")
            if not model_kwargs["use_cache"]:
                raise ValueError("assisted generate requires `use_cache=True`")

            # 11. Get the candidate generator, given the parameterization
            candidate_generator = self._get_candidate_generator(
                generation_config=generation_config,
                input_ids=input_ids,
                inputs_tensor=inputs_tensor,
                assistant_model=assistant_model,
                logits_processor=logits_processor,
                model_kwargs=model_kwargs,
            )

            # 12. run assisted generate
            return self.assisted_decoding(
                input_ids,
                candidate_generator=candidate_generator,
                do_sample=generation_config.do_sample,
                logits_processor=prepared_logits_processor,
                logits_warper=self._get_logits_warper(generation_config) if generation_config.do_sample else None,
                stopping_criteria=prepared_stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                output_logits=generation_config.output_logits,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )
        if generation_mode == GenerationMode.GREEDY_SEARCH:
            # 11. run greedy search
            return self.greedy_search(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                output_logits=generation_config.output_logits,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                lazy_mode=lazy_mode,
                ignore_eos=generation_config.ignore_eos,
                profiling_warmup_steps=profiling_warmup_steps,
                profiling_steps=profiling_steps,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.CONTRASTIVE_SEARCH:
            if not model_kwargs["use_cache"]:
                raise ValueError("Contrastive search requires `use_cache=True`")

            return self.contrastive_search(
                input_ids,
                top_k=generation_config.top_k,
                penalty_alpha=generation_config.penalty_alpha,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                output_logits=generation_config.output_logits,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                sequential=generation_config.low_memory,
                profiling_warmup_steps=profiling_warmup_steps,
                profiling_steps=profiling_steps,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.SAMPLE:
            # 11. prepare logits warper
            logits_warper = self._get_logits_warper(generation_config)

            # 12. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 13. run sample
            return self.sample(
                input_ids,
                logits_processor=prepared_logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=prepared_stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                output_logits=generation_config.output_logits,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                lazy_mode=lazy_mode,
                ignore_eos=generation_config.ignore_eos,
                profiling_warmup_steps=profiling_warmup_steps,
                profiling_steps=profiling_steps,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.BEAM_SEARCH:
            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            return self.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                output_logits=generation_config.output_logits,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                sequential=generation_config.low_memory,
                lazy_mode=lazy_mode,
                profiling_warmup_steps=profiling_warmup_steps,
                profiling_steps=profiling_steps,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.BEAM_SAMPLE:
            # 11. prepare logits warper
            logits_warper = self._get_logits_warper(generation_config)

            # 12. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )

            # 13. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 14. run beam sample
            return self.beam_sample(
                input_ids,
                beam_scorer,
                logits_processor=prepared_logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=prepared_stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                output_logits=generation_config.output_logits,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                lazy_mode=lazy_mode,
                profiling_warmup_steps=profiling_warmup_steps,
                profiling_steps=profiling_steps,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.GROUP_BEAM_SEARCH:
            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                num_beam_groups=generation_config.num_beam_groups,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            return self.group_beam_search(
                input_ids,
                beam_scorer,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                output_logits=generation_config.output_logits,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                lazy_mode=lazy_mode,
                profiling_warmup_steps=profiling_warmup_steps,
                profiling_steps=profiling_steps,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.CONSTRAINED_BEAM_SEARCH:
            final_constraints = []
            if generation_config.constraints is not None:
                final_constraints = generation_config.constraints

            if generation_config.force_words_ids is not None:

                def typeerror():
                    raise ValueError(
                        "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]` "
                        f"of positive integers, but is {generation_config.force_words_ids}."
                    )

                if (
                    not isinstance(generation_config.force_words_ids, list)
                    or len(generation_config.force_words_ids) == 0
                ):
                    typeerror()

                for word_ids in generation_config.force_words_ids:
                    if isinstance(word_ids[0], list):
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any(not isinstance(token_ids, list) for token_ids in word_ids):
                            typeerror()
                        if any(
                            any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
                            for token_ids in word_ids
                        ):
                            typeerror()

                        constraint = DisjunctiveConstraint(word_ids)
                    else:
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
                            typeerror()

                        constraint = PhrasalConstraint(word_ids)
                    final_constraints.append(constraint)

            # 11. prepare beam search scorer
            constrained_beam_scorer = ConstrainedBeamSearchScorer(
                constraints=final_constraints,
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            return self.constrained_beam_search(
                input_ids,
                constrained_beam_scorer=constrained_beam_scorer,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                output_logits=generation_config.output_logits,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                lazy_mode=lazy_mode,
                profiling_warmup_steps=profiling_warmup_steps,
                profiling_steps=profiling_steps,
                **model_kwargs,
            )

    @torch.no_grad()
    def contrastive_search(
        self,
        input_ids: torch.LongTensor,
        top_k: Optional[int] = 1,
        penalty_alpha: Optional[float] = 0,
        logits_processor: Optional[LogitsProcessorList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        sequential: Optional[bool] = None,
        lazy_mode: Optional[bool] = False,
        profiling_warmup_steps: Optional[int] = 0,
        profiling_steps: Optional[int] = 0,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""Generates sequences of token ids for models with a language modeling head using **contrastive search**
        and can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin.contrastive_search`] directly. Use
        generate() instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            top_k (`int`, *optional*, defaults to 1):
                The size of the candidate set that is used to re-rank for contrastive search
            penalty_alpha (`float`, *optional*, defaults to 0):
                The degeneration penalty for contrastive search; activate when it is larger than 0
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            output_logits (`bool`, *optional*, defaults to `False`):
                Whether or not to return the raw prediction logit scores. See `logits` under returned tensors
                for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`transformers.generationutils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            lazy_mode (`bool`, *optional*, defaults to `False`):
                Whether the run is executed in lazy mode or not (i.e. eager mode).
            profiling_warmup_steps (`int`, *optional*, defaults to 0):
                Number of steps to ignore for profiling.
            profiling_steps (`int`, *optional*, defaults to 0):
                Number of steps to be captured when enabling profiling.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`transformers.generation.GenerateDecoderOnlyOutput`],
            [`transformers.generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`: A `torch.LongTensor`
            containing the generated tokens (default behaviour) or a
            [`transformers.generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`transformers.generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:
        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )

        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        >>> model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        >>> # set pad_token_id to eos_token_id because OPT does not have a PAD token
        >>> model.config.pad_token_id = model.config.eos_token_id
        >>> input_prompt = "DeepMind Company is"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt")
        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=64)])
        >>> outputs = model.contrastive_search(
        ...     **input_ids, penalty_alpha=0.6, top_k=4, stopping_criteria=stopping_criteria
        ... )
        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['DeepMind Company is a company that focuses on the development and \
          commercialization of artificial intelligence (AI). \
          DeepMind’s mission is to help people understand and solve problems \
          that are difficult to solve in the world today.\n\n\
          In this post, we talk about the benefits of deep learning in business and how it']
        ```
        """

        raise NotImplementedError("Contrastive search is not supported by optimum-habana yet.")

    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        lazy_mode: Optional[bool] = False,
        ignore_eos: Optional[bool] = False,
        profiling_warmup_steps: Optional[int] = 0,
        profiling_steps: Optional[int] = 0,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""Generates sequences of token ids for models with a language modeling head using **greedy decoding** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin.greedy_search`] directly. Use generate()
        instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>


        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            output_logits (`bool`, *optional*, defaults to `False`):
                Whether or not to return the raw prediction logit scores. See `logits` under returned tensors
                for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`transformers.generationutils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            lazy_mode (`bool`, *optional*, defaults to `False`):
                Whether the run is executed in lazy mode or not (i.e. eager mode).
            ignore_eos (`bool`, *optional*, defaults to `False`):
                Whether to ignore finished sequences (faster in lazy mode and with HPU graphs) or not (eager mode).
            profiling_warmup_steps (`int`, *optional*, defaults to 0):
                Number of steps to ignore for profiling.
            profiling_steps (`int`, *optional*, defaults to 0):
                Number of steps to be captured when enabling profiling.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`transformers.generation.GenerateDecoderOnlyOutput`],
            [`transformers.generation.GenerateEncoderDecoderOutput`]
            or `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`transformers.generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`transformers.generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )

        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
        >>> model.generation_config.pad_token_id = model.generation_config.eos_token_id

        >>> input_prompt = "It might be possible to"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
        ...     ]
        ... )
        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

        >>> outputs = model.greedy_search(
        ...     input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ["It might be possible to get a better understanding of the nature of the problem, but it's not"]
        ```
        """
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                (
                    "`max_length` is deprecated in this function, use"
                    " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead."
                ),
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        if not ignore_eos:
            unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        hb_profer = HabanaProfile(warmup=profiling_warmup_steps, active=profiling_steps)
        hb_profer.start()
        this_peer_finished = False  # used by synced_gpus only
        bucket_size = model_kwargs.get("bucket_size", -1)
        prev_idx = -1  # avoiding calculate cache_idx when its value is not changing
        bucket_internal = model_kwargs.get("bucket_internal", None)
        reduce_recompile = model_kwargs.get("reduce_recompile", False)

        prompt_len = input_ids.shape[-1]
        if not bucket_internal:
            if bucket_size >= 0:
                inc = iter(incrementor(bucket_size, prompt_len))
            if bucket_size > 0:
                assert "position_ids" not in model_kwargs, "Untested path"
        cur_len = prompt_len
        token_idx = model_kwargs.get("token_idx", None)
        if token_idx is not None:
            # Update cur_len in case of static shapes
            cur_len = token_idx.item()
        while True:
            if lazy_mode:
                self.htcore_generation.mark_step()

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            if bucket_size > 0 and not bucket_internal:
                # it will not have been padded if bucket_size > 0
                params = next(inc)
                input_ids, model_kwargs = self.update_model_kwargs_for_bucketing(
                    params, input_ids, model_kwargs, pad_token_id, bucket_size, reduce_recompile
                )

            # prepare model inputs
            model_kwargs["lazy_mode"] = lazy_mode
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            hpu_graphs_kwargs = self._get_hpu_graphs_kwargs(model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                **hpu_graphs_kwargs,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            token_idx = model_kwargs.get("token_idx", None)
            if token_idx is not None and outputs.logits.shape[-2] > 1:
                # case1 (w/o KV caching): outputs.logits.shape: [batch_size, max_length, vocab_size]
                if self.config.is_encoder_decoder:
                    next_token_logits = outputs.logits[:, token_idx - 1, :]
                    next_tokens_scores = logits_processor(input_ids[:, :token_idx], next_token_logits)
                else:
                    next_token_logits = torch.index_select(outputs.logits, -2, token_idx - 1).squeeze(-2)
                    next_tokens_scores = logits_processor(input_ids, next_token_logits)
            else:
                next_token_logits = outputs.logits[:, -1, :]
                if token_idx is not None and self.config.is_encoder_decoder:
                    # case2 (with KV caching): outputs.logits.shape: [batch_size, 1, vocab_size]
                    next_tokens_scores = logits_processor(input_ids[:, :token_idx], next_token_logits)
                else:
                    # case3 (default case): token_idx is None
                    next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)
            # finished sentences should have their next token be a padding token
            if not ignore_eos and eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            if token_idx is not None:
                input_ids.index_copy_(
                    1, token_idx, next_tokens.unsqueeze(-1) if next_tokens.dim() == 1 else next_tokens
                )
            else:
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if bucket_size > 0 and bucket_internal:
                # Calculate slice idx for kv cache during the decode phase.
                # Breaking down the kv cache in the attention block helps to reduce computation time.
                if model_kwargs.get("token_idx_cpu") <= (model_kwargs["kv_cache_len"] // bucket_size) * bucket_size:
                    idx = (model_kwargs.get("token_idx_cpu") - 1) // bucket_size
                    if prev_idx != idx:
                        model_kwargs["cache_idx"] = (idx + 1) * bucket_size
                        prev_idx = idx
                else:
                    model_kwargs["cache_idx"] = model_kwargs["kv_cache_len"]
            cur_len = cur_len + 1

            # if eos_token was found in one sentence, set sentence to finished
            if not ignore_eos and eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )
                # stop when each sentence is finished
                if not ignore_eos and unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores, token_idx=cur_len):
                this_peer_finished = True

            hb_profer.step()

            if this_peer_finished and not synced_gpus:
                break

        hb_profer.stop()
        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids

    def sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        lazy_mode: Optional[bool] = False,
        ignore_eos: Optional[bool] = False,
        profiling_warmup_steps: Optional[int] = 0,
        profiling_steps: Optional[int] = 0,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin.sample`] directly. Use generate() instead.
        For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            output_logits (`bool`, *optional*, defaults to `False`):
                Whether or not to return the raw prediction logit scores. See `logits` under returned tensors for
                more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`transformers.generationutils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            lazy_mode (`bool`, *optional*, defaults to `False`):
                Whether the run is executed in lazy mode or not (i.e. eager mode).
            ignore_eos (`bool`, *optional*, defaults to `False`):
                Whether to ignore finished sequences (faster in lazy mode and with HPU graphs) or not (eager mode).
            profiling_warmup_steps (`int`, *optional*, defaults to 0):
                Number of steps to ignore for profiling.
            profiling_steps (`int`, *optional*, defaults to 0):
                Number of steps to be captured when enabling profiling.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`transformers.generation.GenerateDecoderOnlyOutput`],
            [`transformers.generation.GenerateEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`transformers.generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`transformers.generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     TopKLogitsWarper,
        ...     TemperatureLogitsWarper,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
        >>> model.config.pad_token_id = model.config.eos_token_id
        >>> model.generation_config.pad_token_id = model.config.eos_token_id

        >>> input_prompt = "Today is a beautiful day, and"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(15, eos_token_id=model.generation_config.eos_token_id),
        ...     ]
        ... )
        >>> # instantiate logits processors
        >>> logits_warper = LogitsProcessorList(
        ...     [
        ...         TopKLogitsWarper(50),
        ...         TemperatureLogitsWarper(0.7),
        ...     ]
        ... )

        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

        >>> torch.manual_seed(0)  # doctest: +IGNORE_RESULT
        >>> outputs = model.sample(
        ...     input_ids,
        ...     logits_processor=logits_processor,
        ...     logits_warper=logits_warper,
        ...     stopping_criteria=stopping_criteria,
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Today is a beautiful day, and we must do everything possible to make it a day of celebration.']
        ```
        """

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                (
                    "`max_length` is deprecated in this function, use"
                    " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                ),
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_logits = output_logits if output_logits is not None else self.generation_config.output_logits
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        # TODO: no ignore_eos check here since there is a compilation error, will add ignore_eos here if fixed
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        hb_profer = HabanaProfile(warmup=profiling_warmup_steps, active=profiling_steps)
        hb_profer.start()
        this_peer_finished = False  # used by synced_gpus only
        cur_len = input_ids.shape[-1]
        token_idx = model_kwargs.get("token_idx", None)
        if token_idx is not None:
            # Update cur_len in case of static shapes
            cur_len = token_idx.item()

        # auto-regressive generation
        while True:
            if lazy_mode:
                self.htcore_generation.mark_step()

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_kwargs["lazy_mode"] = lazy_mode
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            hpu_graphs_kwargs = self._get_hpu_graphs_kwargs(model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                **hpu_graphs_kwargs,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            token_idx = model_kwargs.get("token_idx", None)
            if token_idx is not None and outputs.logits.shape[-2] > 1:
                next_token_logits = torch.index_select(outputs.logits, -2, token_idx - 1).squeeze(-2)
            else:
                next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            # TODO: no ignore_eos check here since there is a compilation error, will add ignore_eos here if fixed
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            if token_idx is not None:
                input_ids.index_copy_(
                    1, token_idx, next_tokens.unsqueeze(-1) if next_tokens.dim() == 1 else next_tokens
                )
            else:
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            cur_len = cur_len + 1
            # if eos_token was found in one sentence, set sentence to finished
            if not ignore_eos and eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if not ignore_eos and unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores, token_idx=cur_len):
                this_peer_finished = True

            hb_profer.step()

            if this_peer_finished and not synced_gpus:
                break

        hb_profer.stop()
        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids

    def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        sequential: Optional[bool] = None,
        lazy_mode: Optional[bool] = False,
        profiling_warmup_steps: Optional[int] = 0,
        profiling_steps: Optional[int] = 0,
        **model_kwargs,
    ) -> Union[GenerateBeamOutput, torch.LongTensor]:
        r"""Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin.beam_search`] directly. Use generate()
        instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            beam_scorer (`BeamScorer`):
                An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            output_logits (`bool`, *optional*, defaults to `False`):
                Whether or not to return the raw prediction logit scores. See `logits` under returned tensors for
                more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`transformers.generationutils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            sequential (`bool`, defaults to `False`):
                By default, beam search has `batch_size * num_beams` as effective batch size (see `beam_search()` for
                more details). This flag will avoid parallelizing the beam search and will instead run beam search
                sequentially.
            lazy_mode (`bool`, *optional*, defaults to `False`):
                Whether the run is executed in lazy mode or not (i.e. eager mode).
            profiling_warmup_steps (`int`, *optional*, defaults to 0):
                Number of steps to ignore for profiling.
            profiling_steps (`int`, *optional*, defaults to 0):
                Number of steps to be captured when enabling profiling.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`transformers.generation.utils.GenerateBeamDecoderOnlyOutput`],
            [`transformers.generation.GenerateBeamEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`transformers.generation.GenerateBeamDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`transformers.generation.GenerateBeamEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForSeq2SeqLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     BeamSearchScorer,
        ... )
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

        >>> encoder_input_str = "translate English to German: How old are you?"
        >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids

        >>> # lets run beam search using 3 beams
        >>> num_beams = 3
        >>> # define decoder start token ids
        >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
        >>> input_ids = input_ids * model.config.decoder_start_token_id

        >>> # add encoder_outputs to model keyword arguments
        >>> model_kwargs = {
        ...     "encoder_outputs": model.get_encoder()(
        ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
        ...     )
        ... }

        >>> # instantiate beam scorer
        >>> beam_scorer = BeamSearchScorer(
        ...     batch_size=1,
        ...     num_beams=num_beams,
        ...     device=model.device,
        ... )

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
        ...     ]
        ... )

        >>> outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Wie alt bist du?']
        ```
        """
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        sequential = sequential if sequential is not None else self.generation_config.low_memory
        if max_length is not None:
            warnings.warn(
                (
                    "`max_length` is deprecated in this function, use"
                    " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                ),
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_logits = output_logits if output_logits is not None else self.generation_config.output_logits
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape
        token_idx = model_kwargs.get("token_idx", None)
        if token_idx is not None:
            # Update cur_len in case of static shapes
            cur_len = token_idx.item()

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        if self.generation_config.static_shapes:
            beam_trace_scores = torch.zeros(
                (input_ids.shape[1], 2 * batch_size * num_beams), device=input_ids.device, dtype=torch.float32
            )
            beam_trace_indices = torch.zeros(
                (input_ids.shape[1], 2 * batch_size * num_beams), device=input_ids.device, dtype=torch.int64
            )
            beam_trace_tokens = torch.zeros(
                (input_ids.shape[1], 2 * batch_size * num_beams), device=input_ids.device, dtype=torch.int64
            )
            beam_trace_idx = torch.tensor(0, device=input_ids.device)
            num_eos_tokens = torch.zeros((1), device=input_ids.device, dtype=torch.int64)
            num_beams_tensor = torch.tensor(num_beams, device=input_ids.device, dtype=torch.int64)

        def finalize_beams(initial_ids, beam_trace, model_config, length_penalty):
            beam_trace_idx, beam_trace_scores, beam_trace_indices, beam_trace_tokens = beam_trace
            bs = initial_ids.shape[0]
            num_beams = beam_trace_scores.shape[1] // (2 * bs)

            beam_trace_idx = beam_trace_idx.item()
            beam_trace_scores = beam_trace_scores[:beam_trace_idx, :]
            beam_trace_indices = beam_trace_indices[:beam_trace_idx, :]
            beam_trace_tokens = beam_trace_tokens[:beam_trace_idx, :]

            # (score, parent_beam, token_id, is_finished)
            root = (float("-inf"), None, None, False)

            def resolve_beam(beam):
                if beam == root:
                    return []
                score, prev, tok, is_finished = beam
                rest = resolve_beam(prev)
                rest.append(tok)
                return rest

            prev_beams = [[root] * num_beams] * bs
            best = [root] * bs

            def beam_score(beam):
                return (beam[3], beam[0])

            for step, (scores, indices, tokens) in enumerate(
                zip(beam_trace_scores, beam_trace_indices, beam_trace_tokens)
            ):
                cur_beams = [[] for _ in range(bs)]
                for idx, (s, i, t) in enumerate(zip(scores, indices, tokens)):
                    batch = idx // (num_beams * 2)
                    idx = idx % (num_beams * 2)
                    b_len = 1 + step
                    b_score = s.item() / (b_len**length_penalty)
                    b_tok = t.item()
                    is_finished = b_tok == model_config.eos_token_id
                    if len(cur_beams[batch]) >= num_beams:
                        continue
                    beam = (b_score, prev_beams[batch][i], b_tok, is_finished)
                    if not is_finished:
                        cur_beams[batch].append(beam)
                    if is_finished or (step + 1 == beam_trace_idx):
                        if beam_score(best[batch]) < beam_score(beam):
                            best[batch] = beam
                prev_beams = cur_beams

            def expand_if_needed(tensor, new_size, value, dim=-1):
                orig_len = tensor.shape[dim]
                padding_len = new_size - orig_len
                import torch.nn.functional as F

                if padding_len > 0:
                    if dim == -1:
                        return F.pad(tensor, (0, padding_len), value=value)
                    elif dim == -2:
                        return F.pad(tensor, (0, 0, 0, padding_len), value=value)
                    else:
                        assert False, f"Unsupported dim value: {dim}"
                return tensor

            result = [
                torch.cat(
                    [initial_ids[i], torch.tensor(resolve_beam(b), dtype=initial_ids.dtype, device=initial_ids.device)]
                )
                for i, b in enumerate(best)
            ]
            max_length = max([t.shape[-1] for t in result])
            result = [expand_if_needed(res, max_length, model_config.pad_token_id) for res in result]
            input_ids = torch.stack(result)
            return input_ids

        hb_profer = HabanaProfile(warmup=profiling_warmup_steps, active=profiling_steps)
        hb_profer.start()
        this_peer_finished = False  # used by synced_gpus only

        bucket_size = model_kwargs.get("bucket_size", -1)
        reduce_recompile = model_kwargs.get("reduce_recompile", False)
        prompt_len = input_ids.shape[-1]
        if bucket_size >= 0:
            inc = iter(incrementor(bucket_size, prompt_len))
        if bucket_size > 0:
            assert "position_ids" not in model_kwargs, "Untested path"
        if self.generation_config.static_shapes:
            initial_ids = input_ids[::num_beams, 0:cur_len]
        while True:
            if lazy_mode:
                self.htcore_generation.mark_step()
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            if bucket_size > 0:
                # it will not have been padded if bucket_size > 0
                params = next(inc)
                input_ids, model_kwargs = self.update_model_kwargs_for_bucketing(
                    params, input_ids, model_kwargs, pad_token_id, bucket_size, reduce_recompile
                )

            model_kwargs["lazy_mode"] = lazy_mode
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # if sequential is True, split the input to batches of batch_size and run sequentially
            if sequential:
                if any(
                    model_name in self.__class__.__name__.lower()
                    for model_name in [
                        "fsmt",
                        "reformer",
                        "bloom",
                        "ctrl",
                        "gpt_bigcode",
                        "transo_xl",
                        "xlnet",
                        "cpm",
                    ]
                ):
                    raise RuntimeError(
                        f"Currently generation for {self.__class__.__name__} is not supported "
                        f"for `low_memory beam_search`. Please open an issue on GitHub if you need this feature."
                    )

                inputs_per_sub_batches = _split_model_inputs(
                    model_inputs, split_size=batch_size, full_batch_size=batch_beam_size
                )
                outputs_per_sub_batch = [
                    self(
                        **inputs_per_sub_batch,
                        return_dict=True,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                    )
                    for inputs_per_sub_batch in inputs_per_sub_batches
                ]

                outputs = stack_model_outputs(outputs_per_sub_batch)
            else:
                hpu_graphs_kwargs = self._get_hpu_graphs_kwargs(model_kwargs)
                outputs = self(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    **hpu_graphs_kwargs,
                )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            token_idx = model_kwargs.get("token_idx", None)
            if token_idx is not None and outputs.logits.shape[-2] > 1:
                next_token_logits = torch.index_select(outputs.logits, -2, token_idx - 1).squeeze(-2)
            else:
                next_token_logits = outputs.logits[:, -1, :]

            next_token_scores = torch.nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            if token_idx is not None:
                next_token_scores_processed = logits_processor(input_ids[:, :token_idx], next_token_scores)
            else:
                next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
                next_token_scores_processed
            )

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Sample 1 + len(eos_token_id) next tokens for each beam so we have at least 1 non eos token per beam.
            n_eos_tokens = len(eos_token_id) if eos_token_id else 0
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, max(2, 1 + n_eos_tokens) * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            if self.generation_config.static_shapes:
                beam_scores = next_token_scores.flatten()
                static_beam_indices = next_indices.flatten()

                beam_tokens = next_tokens.remainder(vocab_size).flatten()

                beam_trace_scores.index_copy_(0, beam_trace_idx, beam_scores.unsqueeze(0))
                beam_trace_indices.index_copy_(0, beam_trace_idx, static_beam_indices.unsqueeze(0))
                beam_trace_tokens.index_copy_(0, beam_trace_idx, beam_tokens.unsqueeze(0))
                beam_trace_idx.add_(1)

                if self.generation_config.early_stopping:
                    num_eos_tokens.add_(beam_tokens[0:num_beams].eq(self.config.eos_token_id).sum())

                beam_scores.add_(torch.where(beam_tokens.eq(self.config.eos_token_id), float("-inf"), 0.0))
                beam_scores = beam_scores.view(batch_size, -1).unsqueeze(0)
                _, selected = torch.topk(beam_scores, k=num_beams, dim=-1, largest=True, sorted=True)
                offset = torch.arange(0, torch.numel(beam_scores), beam_scores.shape[-1]).unsqueeze(-1)
                selected = (selected + offset).flatten()
                beam_scores = beam_scores.flatten().index_select(0, selected)
                beam_tokens = beam_tokens.index_select(0, selected)
                static_beam_indices = static_beam_indices.index_select(0, selected)

                prev_beams = outputs.logits.shape[0] // batch_size

                beam_offsets = torch.arange(0, 1, prev_beams, dtype=torch.int32)
                beam_offsets = beam_offsets.to(device=outputs.logits.device)
                static_beam_indices = (static_beam_indices.view(batch_size, -1) + beam_offsets.unsqueeze(-1)).flatten()

                next_tokens = beam_tokens.unsqueeze(-1)
                beam_next_tokens = next_tokens
                beam_idx = static_beam_indices
            else:
                next_tokens = next_tokens % vocab_size
                # stateless
                beam_outputs = beam_scorer.process(
                    input_ids,
                    next_token_scores,
                    next_tokens,
                    next_indices,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    beam_indices=beam_indices,
                    decoder_prompt_len=prompt_len,
                )
                beam_scores = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]

            if token_idx is not None:
                input_ids = torch.index_select(input_ids, 0, beam_idx)
                input_ids.index_copy_(
                    1, token_idx, beam_next_tokens.unsqueeze(-1) if beam_next_tokens.dim() == 1 else beam_next_tokens
                )
            else:
                input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past_key_values"] is not None:
                if model_kwargs["reuse_cache"]:
                    model_kwargs["past_key_values"] = unwrap_deepspeed_model(self).reorder_kv_cache(beam_idx)
                else:
                    model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                        model_kwargs["past_key_values"], beam_idx
                    )

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            hb_profer.step()
            if self.generation_config.static_shapes:
                is_min_length_reached = (
                    self.generation_config.min_length and cur_len >= self.generation_config.min_length
                )
                if (
                    self.generation_config.early_stopping
                    and is_min_length_reached
                    and num_eos_tokens >= num_beams_tensor
                ):
                    break
                elif stopping_criteria(input_ids, scores, token_idx=cur_len):
                    break
            elif stopping_criteria(input_ids, scores) or (beam_scorer.is_done and not lazy_mode):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True
        hb_profer.stop()

        if self.generation_config.static_shapes:
            beam_trace = (beam_trace_idx, beam_trace_scores, beam_trace_indices, beam_trace_tokens)
            from collections import UserDict

            def map_tensors(obj, fn):
                constructor = type(obj)
                if isinstance(obj, tuple):
                    return constructor(map_tensors(v, fn) for v in obj)
                if isinstance(obj, list):
                    return constructor([map_tensors(v, fn) for v in obj])
                if isinstance(obj, dict) or isinstance(obj, UserDict):
                    return constructor({k: map_tensors(v, fn) for k, v in obj.items()})
                if isinstance(obj, torch.Tensor):
                    return fn(obj)
                return obj

            def move(obj, device):
                return map_tensors(obj, lambda t: t.to(device))

            sequence_outputs = {}
            sequence_outputs["sequences"] = finalize_beams(
                initial_ids.cpu(), move(beam_trace, "cpu"), self.config, self.generation_config.length_penalty
            )
        else:
            sequence_outputs = beam_scorer.finalize(
                input_ids,
                beam_scores,
                next_tokens,
                beam_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                max_length=stopping_criteria.max_length,
                beam_indices=beam_indices,
                decoder_prompt_len=prompt_len,
            )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return GenerateBeamEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateBeamDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return sequence_outputs["sequences"]

    def beam_sample(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        lazy_mode: Optional[bool] = False,
        profiling_warmup_steps: Optional[int] = 0,
        profiling_steps: Optional[int] = 0,
        **model_kwargs,
    ) -> Union[GenerateBeamOutput, torch.LongTensor]:
        r"""Generates sequences of token ids for models with a language modeling head using **beam search multinomial
        sampling** and can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin.beam_sample`] directly. Use generate()
        instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            beam_scorer (`BeamScorer`):
                A derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            output_logits (`bool`, *optional*, defaults to `False`):
                Whether or not to return the raw prediction logit scores. See `logits` under returned tensors for
                more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`transformers.generationutils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            lazy_mode (`bool`, *optional*, defaults to `False`):
                Whether the run is executed in lazy mode or not (i.e. eager mode).
            profiling_warmup_steps (`int`, *optional*, defaults to 0):
                Number of steps to ignore for profiling.
            profiling_steps (`int`, *optional*, defaults to 0):
                Number of steps to be captured when enabling profiling.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`transformers.generation.GenerateBeamDecoderOnlyOutput`],
            [`transformers.generation.GenerateBeamEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`transformers.generation.GenerateBeamDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`transformers.generation.GenerateBeamEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForSeq2SeqLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     TopKLogitsWarper,
        ...     TemperatureLogitsWarper,
        ...     BeamSearchScorer,
        ... )
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

        >>> encoder_input_str = "translate English to German: How old are you?"
        >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids

        >>> # lets run beam search using 3 beams
        >>> num_beams = 3
        >>> # define decoder start token ids
        >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
        >>> input_ids = input_ids * model.config.decoder_start_token_id

        >>> # add encoder_outputs to model keyword arguments
        >>> model_kwargs = {
        ...     "encoder_outputs": model.get_encoder()(
        ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
        ...     )
        ... }

        >>> # instantiate beam scorer
        >>> beam_scorer = BeamSearchScorer(
        ...     batch_size=1,
        ...     max_length=model.config.max_length,
        ...     num_beams=num_beams,
        ...     device=model.device,
        ... )

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id)]
        ... )
        >>> # instantiate logits processors
        >>> logits_warper = LogitsProcessorList(
        ...     [
        ...         TopKLogitsWarper(50),
        ...         TemperatureLogitsWarper(0.7),
        ...     ]
        ... )

        >>> outputs = model.beam_sample(
        ...     input_ids, beam_scorer, logits_processor=logits_processor, logits_warper=logits_warper, **model_kwargs
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Wie alt bist du?']
        ```
        """

        raise NotImplementedError("Beam search sampling is not supported by optimum-habana yet.")

    def group_beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        lazy_mode: Optional[bool] = False,
        profiling_warmup_steps: Optional[int] = 0,
        profiling_steps: Optional[int] = 0,
        **model_kwargs,
    ):
        r"""Generates sequences of token ids for models with a language modeling head using **diverse beam search
        decoding** and can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin.group_beam_search`] directly. Use
        generate() instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            beam_scorer (`BeamScorer`):
                An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            output_logits (`bool`, *optional*, defaults to `False`):
                Whether or not to return the raw prediction logit scores. See `logits` under returned tensors for
                more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`transformers.generationutils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            lazy_mode (`bool`, *optional*, defaults to `False`):
                Whether the run is executed in lazy mode or not (i.e. eager mode).
            profiling_warmup_steps (`int`, *optional*, defaults to 0):
                Number of steps to ignore for profiling.
            profiling_steps (`int`, *optional*, defaults to 0):
                Number of steps to be captured when enabling profiling.
            model_kwargs:
                Additional model specific kwargs that will be forwarded to the `forward` function of the model. If
                model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`transformers.generation.GenerateBeamDecoderOnlyOutput`],
            [`transformers.generation.GenerateBeamEncoderDecoderOutput`] or
            `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`transformers.generation.GenerateBeamDecoderOnlyOutput`] if
            [`transformers.generation.BeamSearchDecoderOnlyOutput`] if
            `model.config.is_encoder_decoder=False` and `return_dict_in_generate=True` or
            a [`transformers.generation.GenerateBeamEncoderDecoderOutput`]
            if `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForSeq2SeqLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     HammingDiversityLogitsProcessor,
        ...     BeamSearchScorer,
        ... )
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

        >>> encoder_input_str = "translate English to German: How old are you?"
        >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids

        >>> # lets run diverse beam search using 6 beams
        >>> num_beams = 6
        >>> # define decoder start token ids
        >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
        >>> input_ids = input_ids * model.config.decoder_start_token_id

        >>> # add encoder_outputs to model keyword arguments
        >>> model_kwargs = {
        ...     "encoder_outputs": model.get_encoder()(
        ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
        ...     )
        ... }

        >>> # instantiate beam scorer
        >>> beam_scorer = BeamSearchScorer(
        ...     batch_size=1,
        ...     max_length=model.config.max_length,
        ...     num_beams=num_beams,
        ...     device=model.device,
        ...     num_beam_groups=3,
        ... )

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         HammingDiversityLogitsProcessor(5.5, num_beams=6, num_beam_groups=3),
        ...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
        ...     ]
        ... )

        >>> outputs = model.group_beam_search(
        ...     input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Wie alt bist du?']
        ```
        """

        raise NotImplementedError("Group beam search is not supported by optimum-habana yet.")

    def constrained_beam_search(
        self,
        input_ids: torch.LongTensor,
        constrained_beam_scorer: ConstrainedBeamSearchScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = None,
        lazy_mode: Optional[bool] = False,
        profiling_warmup_steps: Optional[int] = 0,
        profiling_steps: Optional[int] = 0,
        **model_kwargs,
    ) -> Union[GenerateBeamOutput, torch.LongTensor]:
        r"""Generates sequences of token ids for models with a language modeling head using **constrained beam search
        decoding** and can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin.constrained_beam_search`] directly. Use
        generate() instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            constrained_beam_scorer (`ConstrainedBeamSearchScorer`):
                A derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation, while satisfying a list of positive constraints. For more information, the
                documentation of [`ConstrainedBeamSearchScorer`] should be read.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            output_logits (`bool`, *optional*, defaults to `False`):
                Whether or not to return the raw prediction logit scores. See `logits` under returned tensors for
                more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`transformers.generationutils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            lazy_mode (`bool`, *optional*, defaults to `False`):
                Whether the run is executed in lazy mode or not (i.e. eager mode).
            profiling_warmup_steps (`int`, *optional*, defaults to 0):
                Number of steps to ignore for profiling.
            profiling_steps (`int`, *optional*, defaults to 0):
                Number of steps to be captured when enabling profiling.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`transformers.generation.utils.GenerateBeamDecoderOnlyOutput`],
            [`transformers.generation.GenerateBeamEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`transformers.generation.GenerateBeamDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`transformers.generation.GenerateBeamEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForSeq2SeqLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     ConstrainedBeamSearchScorer,
        ...     PhrasalConstraint,
        ... )
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

        >>> encoder_input_str = "translate English to German: How old are you?"
        >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids

        >>> # lets run beam search using 3 beams
        >>> num_beams = 3
        >>> # define decoder start token ids
        >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
        >>> input_ids = input_ids * model.config.decoder_start_token_id

        >>> # add encoder_outputs to model keyword arguments
        >>> model_kwargs = {
        ...     "encoder_outputs": model.get_encoder()(
        ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
        ...     )
        ... }

        >>> constraint_str = "Sie"
        >>> constraint_token_ids = tokenizer.encode(constraint_str)[:-1]  # slice to remove eos token
        >>> constraints = [PhrasalConstraint(token_ids=constraint_token_ids)]

        >>> # instantiate beam scorer
        >>> beam_scorer = ConstrainedBeamSearchScorer(
        ...     batch_size=1, num_beams=num_beams, device=model.device, constraints=constraints
        ... )

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
        ...     ]
        ... )

        >>> outputs = model.constrained_beam_search(
        ...     input_ids, beam_scorer, constraints=constraints, logits_processor=logits_processor, **model_kwargs
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Wie alt sind Sie?']
        ```
        """

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_logits = output_logits if output_logits is not None else self.generation_config.output_logits
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        batch_size = len(constrained_beam_scorer._beam_hyps)
        num_beams = constrained_beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape
        token_idx = model_kwargs.get("token_idx", None)
        if token_idx is not None:
            # Update cur_len in case of static shapes
            cur_len = token_idx.item()

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
        decoder_prompt_len = input_ids.shape[-1]  # record the prompt length of decoder

        hb_profer = HabanaProfile(warmup=profiling_warmup_steps, active=profiling_steps)
        hb_profer.start()
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_kwargs["lazy_mode"] = lazy_mode
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            hpu_graphs_kwargs = self._get_hpu_graphs_kwargs(model_kwargs)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                **hpu_graphs_kwargs,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            if token_idx is not None and outputs.logits.shape[-2] > 1:
                next_token_logits = torch.index_select(outputs.logits, -2, token_idx - 1).squeeze(-2)
            else:
                next_token_logits = outputs.logits[:, -1, :]

            next_token_scores = torch.nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)

            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
                next_token_scores_processed
            )

            scores_for_all_vocab = next_token_scores.clone()

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Sample 1 + len(eos_token_id) next tokens for each beam so we have at least 1 non eos token per beam.
            n_eos_tokens = len(eos_token_id) if eos_token_id else 0
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, max(2, 1 + n_eos_tokens) * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = (next_tokens / vocab_size).long()
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = constrained_beam_scorer.process(
                input_ids[:, :cur_len],
                next_token_scores,
                next_tokens,
                next_indices,
                scores_for_all_vocab,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
                decoder_prompt_len=decoder_prompt_len,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            if token_idx is not None:
                input_ids = input_ids[beam_idx, :]
                input_ids.index_copy_(
                    1, token_idx, beam_next_tokens.unsqueeze(-1) if beam_next_tokens.dim() == 1 else beam_next_tokens
                )
            else:
                input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past_key_values"] is not None:
                model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                    model_kwargs["past_key_values"], beam_idx
                )

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            hb_profer.step()

            if constrained_beam_scorer.is_done or stopping_criteria(input_ids, scores, token_idx=cur_len):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        hb_profer.stop()
        sequence_outputs = constrained_beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
            decoder_prompt_len=decoder_prompt_len,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None
            if self.config.is_encoder_decoder:
                return GenerateBeamEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateBeamDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return sequence_outputs["sequences"]

    def assisted_decoding(
        self,
        input_ids: torch.LongTensor,
        assistant_model: Optional["PreTrainedModel"] = None,
        candidate_generator: Optional["CandidateGenerator"] = None,
        do_sample: bool = False,
        logits_processor: Optional[LogitsProcessorList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        lazy_mode: Optional[bool] = False,
        profiling_warmup_steps: Optional[int] = 0,
        profiling_steps: Optional[int] = 0,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""Generates sequences of token ids for models with a language modeling head using **greedy decoding** or
        **sample** (depending on `do_sample`), assisted by candidate sequences. Assisted generation is an example of a
        candidate decoding strategy. Can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text
        models.

        <Tip warning={true}>

        In most cases, you do not need to call [`transformers.generation.GenerationMixin.candidate_decoding`].
        Use generate() instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            candidate_generator (`CandidateGenerator`, *optional*):
                A derived instance of [`CandidateGenerator`] that defines how candidate sequences are generated. For
                more information, the documentation of [`CandidateGenerator`] should be read.
                Only one of `assistant_model` or `candidate_generator` should be passed as input to this function.
            assistant_model (`PreTrainedModel`, *optional*):
                An assistant model that can be used to accelerate generation. The assistant model must have the exact
                same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistant model
                is much faster than running generation with the model you're calling generate from. As such, the
                assistant model should be much smaller.
            do_sample (`bool`, *optional*, defaults to `False`):
                Whether or not to use sampling ; use greedy decoding otherwise.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            output_logits (`bool`, *optional*, defaults to `False`):
                Whether or not to return the raw prediction logit scores. See `logits` under returned tensors for
                more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            lazy_mode (`bool`, *optional*, defaults to `False`):
                Whether the run is executed in lazy mode or not (i.e. eager mode).
            profiling_warmup_steps (`int`, *optional*, defaults to 0):
                Number of steps to ignore for profiling.
            profiling_steps (`int`, *optional*, defaults to 0):
                Number of steps to be captured when enabling profiling.
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`transformers.generation.GenerateDecoderOnlyOutput`],
            [`transformers.generation.GenerateEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`transformers.generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`transformers.generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )

        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> assistant_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        >>> # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
        >>> model.generation_config.pad_token_id = model.generation_config.eos_token_id
        >>> input_prompt = "It might be possible to"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids
        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
        ...     ]
        ... )
        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])
        >>> outputs = model.assisted_decoding(
        ...     input_ids,
        ...     assistant_model=assistant_model,
        ...     logits_processor=logits_processor,
        ...     stopping_criteria=stopping_criteria,
        ... )
        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ["It might be possible to get a better understanding of the nature of the problem, but it's not"]
        ```
        """
        raise NotImplementedError("Assisted decoding is not supported by optimum-habana yet.")
