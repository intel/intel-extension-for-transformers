#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import transformers
from intel_extension_for_transformers.transformers.utils import logger
from tqdm import tqdm
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BatchEncoding, LlamaTokenizer)

from lm_eval import utils
from lm_eval.base import BaseLM

TokenSequence = Union[List[int], torch.LongTensor, torch.Tensor, BatchEncoding]


class HFAutoLM(BaseLM):
    _DEFAULT_MAX_LENGTH = 2048
    AUTO_MODEL_CLASS: transformers.AutoModel = None
    AUTO_STRUCTURE = None
    MODEL_CLASSES = {
        "llama": (AutoModelForCausalLM, LlamaTokenizer),
        "chatglm": (AutoModel, AutoTokenizer),
        "auto": (AUTO_MODEL_CLASS, AutoTokenizer),
    }

    def __init__(
        self,
        model_name_or_path=None,
        model=None,
        tokenizer=None,
        config=None,
        model_type=None,
        trust_remote_code=True,
        device="cpu",
        batch_size=1,
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        add_special_tokens=None,
        max_length=None,
        max_gen_toks: Optional[int] = 256,
        with_greedy=False,
        model_format="torch",
    ):
        super().__init__()

        self.model_format = model_format
        self.model_name_or_path = model_name_or_path
        self.model = model
        self.config = config
        self.model_type = model_type
        self._device = device
        self._batch_size = batch_size
        self._dtype = dtype
        self._with_greedy = with_greedy
        self._max_length = max_length
        self._add_special_tokens = add_special_tokens
        self._max_gen_toks = max_gen_toks
        self.num_beam = 1 if with_greedy else 4

        if self.model_format == "torch" and model is None:
            logger.error(
                'Args model is necessary when model_format="torch". \n'
                + "the following is the correct usage, \n"
                + "model = HFCausalLM(model=model, tokenizer=tokenizer) or "
                + "model = HFSeq2SeqLM(model=model, tokenizer=tokenizer)"
            )
            exit()

        if self.model_format == "onnx" and model_name_or_path is None:
            logger.error(
                'Args model_name_or_path is necessary when model_format="onnx". \n'
                + "the following is the correct usage, \n"
                + 'model = HFCausalLM(model_name_or_path=model_name_or_path, model_format="onnx") \n'
                + ' or model = HFSeq2SeqLM(model_name_or_path=model_name_or_path,model_format="onnx")'
            )
            exit()

        if self.model_format == "onnx":
            self.model = self._create_onnx_model(self.model_name_or_path)
        if self.config is None and hasattr(self.model, "config"):
            self.config = self.model.config
        if not hasattr(self.model, "config"):
            if self.config is not None:
                setattr(self.model, "config", self.config)
            else:
                logger.warning("Please provide model config.")
        if self.model_type is None and hasattr(self.model, "config") and hasattr(self.model.config, "model_type"):
            self.model_type = self.model.config.model_type
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif model_name_or_path is not None:
            model_type = next(
                (
                    x
                    for x in self.MODEL_CLASSES.keys()
                    if x in model_name_or_path.lower()
                ),
                "auto",
            )
            self.tokenizer = self.MODEL_CLASSES[model_type][1].from_pretrained(
                model_name_or_path, trust_remote_code=trust_remote_code
            )
        if self.AUTO_STRUCTURE == "Causal":
            self.tokenizer.padding_side = "left"
        assert self.tokenizer is not None, "Please set tokenizer parameter in HFCausalLM or HFSeq2SeqLM."

    def _create_onnx_model(self, model_name_or_path):
        assert (
            self.model_format == "onnx"
        ), 'Please set model_format = "onnx" if the backend is onnx.'
        if self.AUTO_STRUCTURE == "Seq2Seq":
            if not os.path.exists(
                os.path.join(model_name_or_path, "encoder_model.onnx")
            ) or (
                not os.path.exists(
                    os.path.join(model_name_or_path, "decoder_model.onnx")
                )
                and not os.path.exists(
                    os.path.join(model_name_or_path, "decoder_model_merged.onnx")
                )
            ):
                raise ValueError(
                    "Please ensure encoder_model.onnx and "
                    "decoder_model(_merged).onnx are under {}.".format(
                        model_name_or_path
                    )
                )

            import onnxruntime as ort
            from optimum.onnxruntime import ORTModelForSeq2SeqLM
            from transformers import PretrainedConfig

            model_config = PretrainedConfig.from_pretrained(model_name_or_path)
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            if os.path.exists(
                os.path.join(model_name_or_path, "decoder_model_merged.onnx")
            ):
                sessions = ORTModelForSeq2SeqLM.load_model(
                    os.path.join(model_name_or_path, "encoder_model.onnx"),
                    os.path.join(model_name_or_path, "decoder_model_merged.onnx"),
                )

                model = ORTModelForSeq2SeqLM(
                    sessions[0],
                    sessions[1],
                    model_config,
                    model_name_or_path,
                    use_cache=True,
                )

            elif os.path.exists(
                os.path.join(model_name_or_path, "decoder_with_past_model.onnx")
            ):
                sessions = ORTModelForSeq2SeqLM.load_model(
                    os.path.join(model_name_or_path, "encoder_model.onnx"),
                    os.path.join(model_name_or_path, "decoder_model.onnx"),
                    os.path.join(model_name_or_path, "decoder_with_past_model.onnx"),
                )

                model = ORTModelForSeq2SeqLM(
                    sessions[0],
                    sessions[1],
                    model_config,
                    model_name_or_path,
                    sessions[2],
                    use_cache=True,
                )
            else:
                sessions = ORTModelForSeq2SeqLM.load_model(  # pylint: disable=E1120
                    os.path.join(model_name_or_path, "encoder_model.onnx"),
                    os.path.join(model_name_or_path, "decoder_model.onnx"),
                )

                model = ORTModelForSeq2SeqLM(
                    sessions[0],
                    sessions[1],
                    model_config,
                    model_name_or_path,
                    use_cache=False,
                    use_io_binding=False,
                )
        else:
            if not os.path.exists(
                os.path.join(model_name_or_path, "decoder_model.onnx")
            ) and not os.path.exists(
                os.path.join(model_name_or_path, "decoder_model_merged.onnx")
            ):
                raise ValueError(
                    "Couldn't find decoder_model.onnx or decoder_model_merged.onnx in {}.".format(
                        model_name_or_path
                    )
                )

            import onnxruntime as ort
            from optimum.onnxruntime import ORTModelForCausalLM
            from transformers import PretrainedConfig

            model_config = PretrainedConfig.from_pretrained(model_name_or_path)
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            if os.path.exists(
                os.path.join(model_name_or_path, "decoder_model_merged.onnx")
            ):
                sessions = ORTModelForCausalLM.load_model(  # pylint: disable=E1123
                    os.path.join(model_name_or_path, "decoder_model_merged.onnx"),
                    session_options=sess_options,
                )
                model = ORTModelForCausalLM(
                    sessions[0],  # pylint: disable=E1121
                    model_config,
                    model_name_or_path,
                    use_cache=True,
                )
            elif os.path.exists(
                os.path.join(model_name_or_path, "decoder_with_past_model.onnx")
            ):
                sessions = ORTModelForCausalLM.load_model(  # pylint: disable=E1123
                    os.path.join(model_name_or_path, "decoder_model.onnx"),
                    os.path.join(model_name_or_path, "decoder_with_past_model.onnx"),
                    session_options=sess_options,
                )
                model = ORTModelForCausalLM(
                    sessions[0],  # pylint: disable=E1121
                    model_config,
                    model_name_or_path,
                    sessions[1],
                    use_cache=True,
                )
            else:
                sessions = ORTModelForCausalLM.load_model(  # pylint: disable=E1123
                    os.path.join(model_name_or_path, "decoder_model.onnx"),
                    session_options=sess_options,
                )
                model = ORTModelForCausalLM(
                    sessions[0],  # pylint: disable=E1121
                    model_config,
                    model_name_or_path,
                    use_cache=False,
                    use_io_binding=False,
                )

        return model

    @property
    def add_special_tokens(self) -> bool:
        """Whether to include special tokens in encoded text. This should be
        determined by whether or not the model was trained with special tokens.
        TODO: Remove these conditionals once HuggingFace supports a way to
        check whether or not an arbitrary model was trained with special tokens.
        """
        if self._add_special_tokens is not None:
            return self._add_special_tokens
        elif self.AUTO_MODEL_CLASS is transformers.AutoModelForCausalLM:
            return False
        elif self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM:
            return True
        else:
            raise ValueError(
                "Could not determine `add_special_tokens` value from the model "
                "class. Set to `True` or `False` depending on whether the model "
                "was pre-trained with special tokens."
            )

    @property
    def eot_token(self) -> str:
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def max_length(self) -> int:
        """Return the maximum sequence length of the model.
        NOTE: Different model configurations have different max sequence length
        attribute names.
            - n_positions: (CTRLConfig, T5Config)
            - max_position_embeddings: (BartConfig, RoFormerConfig)
            - n_ctx: (GPT2Config)
        NOTE: For relative position encoded models you should specify the max
        sequence length of the model in the constructor via `max_length`.
        """
        if self._max_length is not None:
            return self._max_length
        # Try to get the sequence length from the model config.
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.config, attr):
                return getattr(self.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def batch_size(self) -> int:
        # TODO: Add adaptive batch size.
        return self._batch_size  # * gpus

    @property
    def device(self) -> Union[int, str, torch.device]:
        return self._device

    def tok_encode(self, string: str) -> TokenSequence:
        # TODO: Merge `tok_encode_batch` here.
        return self.tokenizer.encode(string, add_special_tokens=self.add_special_tokens)

    def tok_encode_batch(self, strings: List[str]) -> TokenSequence:
        return self.tokenizer(
            strings,
            padding=True,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
        )

    def tok_decode(self, tokens: torch.LongTensor) -> List[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def greedy_until(
        self, requests: List[Tuple[str, Union[List[str], str]]]
    ) -> List[str]:
        def _collate(x):
            tokens = self.tok_encode(x[0])
            return len(tokens), x[0]

        results = []
        reorder = utils.Reorderer(requests, _collate)

        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()   # pylint: disable=E1101
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size

        for chunk in utils.chunks(
            tqdm(reorder.get_reordered(), disable=False),
            self.batch_size if self.batch_size != "auto" else adaptive_batch_size,
        ):
            context = [c[0] for c in chunk]
            request_args = chunk[0][1]
            stop = request_args.get("until", None)
            stop_sequences = stop if isinstance(stop, list) else [stop]
            max_generation_length = request_args.get("max_length", None)

            assert (
                isinstance(max_generation_length, int) or max_generation_length is None
            )
            assert isinstance(stop_sequences, list) or stop_sequences is None

            # TODO: Find a better way to handle stop sequences for 0-shot.
            if stop_sequences is None:
                until = [self.eot_token]
            else:
                until = stop_sequences + [self.eot_token]

            if max_generation_length is None:
                max_tokens = self.max_gen_toks
            else:
                max_tokens = max_generation_length

            token_context = self.tok_encode_batch(context)

            responses = self._model_generate(    # pylint: disable=E1123, E1120
                inputs=token_context,  
                max_tokens=max_tokens,
                stop=until,
            )
            responses = self.tok_decode(responses.tolist())

            for response in responses:
                # Ensure the generated responses do not contain the stop sequences.
                for term in until:
                    response = response.split(term)[0]
                # partial caching
                self.cache_hook.add_partial("greedy_until", (context, until), response)
                results.append(response)
        return reorder.get_original(results)


class HFCausalLM(HFAutoLM):
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM
    AUTO_STRUCTURE = "Causal"

    def _model_call(
        self, inputs: TokenSequence, labels: Optional[TokenSequence] = None
    ) -> TokenSequence:
        if self.model_type != "chatglm":
            pass
        else:
            input_bs, input_len = inputs.shape
            bos = torch.tensor([130001, 130004]).repeat(input_bs, 1)
            inputs = torch.cat((inputs, bos), 1)

        output = (
            self.model(inputs)
            if self.model_format != "onnx"
            else self.model(inputs, torch.ones(inputs.shape, dtype=torch.int64))
        )
        if isinstance(output, tuple):
            return output[0]
        return output["logits"]

    def _model_generate(
        self,
        inputs: transformers.BatchEncoding,
        max_tokens: int,
        stop: Optional[List[str]] = None,
    ) -> TokenSequence:
        # Ensure that the context does not encroach into the `space`
        # for the generation.
        input_ids = inputs["input_ids"][:, self.max_gen_toks - self.max_length :]
        attention_mask = inputs["attention_mask"][
            :, self.max_gen_toks - self.max_length :
        ]
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, input_ids.shape[1], input_ids.shape[0]
        )

        generations = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # GPT style models require the `generate` `max_length` arg to include the
            # context length, so we instead set `max_new_tokens` which is the number
            # of new tokens to generate, excluding the current number of tokens.
            max_new_tokens=max_tokens,
            stopping_criteria=stopping_criteria,
            do_sample=False,
        )
        return utils.select_continuation_from_batch_left_padding(    # pylint: disable=E1101
            generations, max_context_size=inputs["input_ids"].size(1)
        )


class HFSeq2SeqLM(HFAutoLM):

    """Seq2Seq language modeling.
    You can find a set of supported models in the following documentation:
    https://huggingface.co/docs/transformers/main/model_doc/auto#transformers.AutoModelForSeq2SeqLM
    """

    AUTO_MODEL_CLASS = transformers.AutoModelForSeq2SeqLM
    AUTO_STRUCTURE = "Seq2Seq"

    def _model_call(
        self, inputs: TokenSequence, labels: Optional[TokenSequence] = None
    ) -> TokenSequence:
        if self.model_format == "onnx":
            decoder_start_token_id = self.config.decoder_start_token_id
            pad_token_id = self.config.pad_token_id
            shifted_input_ids = labels["input_ids"].new_zeros(labels["input_ids"].shape)
            shifted_input_ids[..., 1:] = labels["input_ids"][..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id
            shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
            return self.model(
                **inputs,
                decoder_input_ids=shifted_input_ids,
                labels=labels["input_ids"],
            )
        else:
            return self.model(**inputs, labels=labels["input_ids"])

    def loglikelihood(
        self, requests: List[Tuple[str, str]]
    ) -> List[Tuple[float, bool]]:
        new_requests = []
        for chunk in utils.chunks(requests, self.batch_size):
            context, continuation = zip(*chunk)

            # Fill empty contexts with the EOT token.
            context = [
                f"{self.eot_token}" if len(text) == 0 else text for text in context
            ]
            context_enc = self.tok_encode_batch(context)
            for key in context_enc:
                context_enc[key] = context_enc[key][:, -self.max_length :]

            # Remove leading whitespace introduced by the default
            # `text_target_separator` since the context and continuation
            # will not be concatenated as a single (decoder) input.
            continuation = [text.lstrip() for text in continuation]
            continuation_enc = self.tok_encode_batch(list(continuation))
            for key in continuation_enc:
                continuation_enc[key] = continuation_enc[key][:, -self.max_length :]

            new_requests.append(
                ((context, continuation), context_enc, continuation_enc)
            )
        return self._loglikelihood_tokens(new_requests)

    def loglikelihood_rolling(self, requests: List[Tuple[str, str]]) -> List[float]:
        loglikelihoods = []
        for (string,) in tqdm(requests):
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )
            contexts, conts = utils.split_and_pad_windows(    # pylint: disable=E1101
                rolling_token_windows,
                pad_token_id=self.eot_token_id,
                max_seq_len=self.max_length,
            )
            # Manually create BatchEncoding tensors with attention masks as
            # expected by `self._model_call` in `self._loglikelihood_tokens`.
            contexts_enc = torch.Tensor(contexts).long()
            contexts_enc = transformers.tokenization_utils_base.BatchEncoding(
                {
                    "input_ids": contexts_enc,
                    "attention_mask": (contexts_enc != self.eot_token_id).long(),
                }
            )
            conts_enc = torch.Tensor(conts).long()
            conts_enc = transformers.tokenization_utils_base.BatchEncoding(
                {
                    "input_ids": conts_enc,
                    "attention_mask": (conts_enc != self.eot_token_id).long(),
                }
            )
            # TODO: Extract out this call so it only gets called once and also
            # somehow figure out partial caching for.
            rolling_token_windows_request = [
                ((contexts, conts), contexts_enc, conts_enc)
            ]
            string_nll = self._loglikelihood_tokens(
                rolling_token_windows_request, disable_tqdm=True
            )
            string_nll = [x[0] for x in string_nll]  # discard is_greedy
            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)
        return loglikelihoods

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], TokenSequence, TokenSequence]],
        disable_tqdm: Optional[bool] = False,
    ) -> List[Tuple[float, bool]]:
        results = []
        for chunk in tqdm(
            requests, total=math.ceil(len(requests)), disable=disable_tqdm
        ):
            cache_keys, inputs_tokens, targets_tokens = chunk
            inputs_tokens = inputs_tokens.to(self.device)
            targets_tokens = targets_tokens.to(self.device)
            outputs = self._model_call(inputs=inputs_tokens, labels=targets_tokens)
            log_softmaxes = F.log_softmax(outputs.logits, dim=-1)

            output_iterator = zip(
                zip(cache_keys[0], cache_keys[1]),
                log_softmaxes,
                targets_tokens["input_ids"],
                targets_tokens["attention_mask"],
            )
            for cache_key, log_softmax, target_tokens, target_mask in output_iterator:
                length = target_mask.sum()
                log_softmax = log_softmax[:length]
                target_tokens = target_tokens[:length]
                greedy_tokens = log_softmax.argmax(dim=-1)
                max_equal = (greedy_tokens == target_tokens).all()
                target_logits = torch.gather(
                    log_softmax, 1, target_tokens.unsqueeze(-1)
                ).squeeze(-1)
                answer = (float(target_logits.sum()), bool(max_equal))
                results.append(answer)
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)
        return results

    def _model_generate(
        self,
        inputs: transformers.BatchEncoding,
        max_tokens: int,
        stop: Optional[List[str]] = None,
    ) -> TokenSequence:
        input_ids = inputs["input_ids"][:, -self.max_length :].to(self.device)
        attention_mask = inputs["attention_mask"][:, -self.max_length :].to(self.device)

        # Generate one token to calculate the number of start tokens prepended to decoder_input_ids
        # (leaving this here in case the below assumption is violated in the future)
        # one_tok_gen = self.model.generate(
        #    input_ids=torch.zeros((1, 1), dtype=torch.int),
        #    min_length=2,
        #    max_new_tokens=1,
        # ).squeeze()
        # initial_decoder_input_length = len(one_tok_gen) - 1

        # Assume that there will always only be one token in the decoder inputs, assumption holds for existing HF models
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, 1, input_ids.shape[0]
        )

        generations = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            stopping_criteria=stopping_criteria,
            do_sample=False,
        )
        return generations


class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
        self,
        sequence: str,
        tokenizer: transformers.PreTrainedTokenizer,
        initial_decoder_input_length: int,
        batch_size: int,
    ):
        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        self.sequence_id_len = len(self.sequence_ids)
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length :][
            :, -self.sequence_id_len :
        ]

        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker


def stop_sequences_criteria(
    tokenizer: transformers.PreTrainedTokenizer,
    stop_sequences: List[str],
    initial_decoder_input_length: int,
    batch_size: int,
) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(
                    sequence, tokenizer, initial_decoder_input_length, batch_size
                )
                for sequence in stop_sequences
            ],
        ]
    )
