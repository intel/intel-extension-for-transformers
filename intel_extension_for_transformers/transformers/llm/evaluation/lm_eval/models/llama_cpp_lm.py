import logging
import time

import requests
from requests.exceptions import RequestException
from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

import llama_cpp
from llama_cpp import Llama, StoppingCriteriaList, LogitsProcessorList
from llama_cpp.llama_grammar import LlamaGrammar
from llama_cpp.llama_types import *
import uuid

from typing import Iterator, List, Optional, Union, Dict
from typing import Literal, Tuple
import numpy as np
from lm_eval.models.utils import (
    Collator,
    clear_torch_cache,
    get_dtype,
    pad_and_concat,
    stop_sequences_criteria,
)
import ctypes

def topk_numpy(arr, k, dim):
    idx = np.argpartition(-arr, kth=k, axis=dim)
    idx = idx.take(indices=range(k), axis=dim)
    val= np.take_along_axis(arr, indices=idx, axis=dim)
    sorted_idx = np.argsort(-val, axis=dim)
    idx = np.take_along_axis(idx, indices=sorted_idx, axis=dim)
    val = np.take_along_axis(val, indices=sorted_idx, axis=dim)
    return val, idx

def _token_to_piece(model, token: int, special: bool = False) -> str:
    assert model.model is not None
    result = (ctypes.c_char * 8)(0)
    n_tokens = llama_cpp.llama_token_to_piece(model.model, token, result, len(result), special)
    if n_tokens < 0:
        result = (ctypes.c_char * -n_tokens)(0)
        check = llama_cpp.llama_token_to_piece(model.model, token, result, len(result), special)
        if check != -n_tokens:
            raise RuntimeError(f"Failed to get piece: token={token}")
    else:
        result = result[:n_tokens]
    return bytes(result)


class CustomLlamaCpp(Llama):

    def detokenize_bpe(
        self, tokens: List[int], prev_tokens: Optional[List[int]] = None
    ) -> bytes:

        assert self._model is not None
        result = b""
        for token in tokens:
            piece = _token_to_piece(self._model, token)
            result += piece
        return result

    def _create_completion(
        self,
        prompt: Union[str, List[int]],
        suffix: Optional[str] = None,
        max_tokens: Optional[int] = 16,
        temperature: float = 0.8,
        top_p: float = 0.95,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        logprobs: Optional[int] = None,
        echo: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        top_k: int = 40,
        stream: bool = False,
        seed: Optional[int] = None,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        model: Optional[str] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        grammar: Optional[LlamaGrammar] = None,
        logit_bias: Optional[Dict[str, float]] = None,
    ) -> Union[
        Iterator[CreateCompletionResponse], Iterator[CreateCompletionStreamResponse]
    ]:

        assert self._ctx is not None
        assert suffix is None or suffix.__class__ is str

        completion_id: str = f"cmpl-{str(uuid.uuid4())}"
        created: int = int(time.time())
        # If prompt is empty, initialize completion with BOS token to avoid
        # detokenization including a space at the beginning of the completion
        completion_tokens: List[int] = [] if len(prompt) > 0 else [self.token_bos()]
        # Add blank space to start of prompt to match OG llama tokenizer
        prompt_tokens: List[int] = (
            (
                self.tokenize(prompt.encode("utf-8"), special=True)
                if prompt != ""
                else [self.token_bos()]
            )
            if isinstance(prompt, str)
            else prompt
        )
        text: bytes = b""
        returned_tokens: int = 0
        stop = (
            stop if isinstance(stop, list) else [stop] if isinstance(stop, str) else []
        )
        model_name: str = model if model is not None else self.model_path


        if "qwen" in model_name.lower():
            prompt_tokens = [self.token_bos()] + prompt_tokens

        # NOTE: This likely doesn't work correctly for the first token in the prompt
        # because of the extra space added to the start of the prompt_tokens
        if logit_bias is not None:
            logit_bias_map = {int(k): float(v) for k, v in logit_bias.items()}

            def logit_bias_processor(
                input_ids: npt.NDArray[np.intc],
                scores: npt.NDArray[np.single],
            ) -> npt.NDArray[np.single]:
                new_scores = np.copy(
                    scores
                )  # Does it make sense to copy the whole array or can we just overwrite the original one?
                for input_id, score in logit_bias_map.items():
                    new_scores[input_id] = score + scores[input_id]
                return new_scores

            _logit_bias_processor = LogitsProcessorList([logit_bias_processor])
            if logits_processor is None:
                logits_processor = _logit_bias_processor
            else:
                logits_processor = logits_processor.extend(_logit_bias_processor)

        if self.verbose:
            self._ctx.reset_timings()

        if len(prompt_tokens) >= self._n_ctx:
            raise ValueError(
                f"Requested tokens ({len(prompt_tokens)}) exceed context window of {llama_cpp.llama_n_ctx(self.ctx)}"
            )

        if max_tokens is None or max_tokens <= 0:
            # Unlimited, depending on n_ctx.
            max_tokens = self._n_ctx - len(prompt_tokens)

        # Truncate max_tokens if requested tokens would exceed the context window
        max_tokens = (
            max_tokens
            if max_tokens + len(prompt_tokens) < self._n_ctx
            else (self._n_ctx - len(prompt_tokens))
        )

        if stop != []:
            stop_sequences = [s.encode("utf-8") for s in stop]
        else:
            stop_sequences = []

        if logprobs is not None and self.context_params.logits_all is False:
            raise ValueError(
                "logprobs is not supported for models created with logits_all=False"
            )

        if self.cache:
            try:
                cache_item = self.cache[prompt_tokens]
                cache_prefix_len = Llama.longest_token_prefix(
                    cache_item.input_ids.tolist(), prompt_tokens
                )
                eval_prefix_len = Llama.longest_token_prefix(
                    self._input_ids.tolist(), prompt_tokens
                )
                if cache_prefix_len > eval_prefix_len:
                    self.load_state(cache_item)
                    if self.verbose:
                        print("Llama._create_completion: cache hit", file=sys.stderr)
            except KeyError:
                if self.verbose:
                    print("Llama._create_completion: cache miss", file=sys.stderr)

        if seed is not None:
            self._ctx.set_rng_seed(seed)

        finish_reason = "length"
        multibyte_fix = 0
        for token in self.generate(
            prompt_tokens,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            typical_p=typical_p,
            temp=temperature,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            grammar=grammar,
        ):
            assert self._model.model is not None
            if llama_cpp.llama_token_is_eog(self._model.model, token):
                text = self.detokenize(completion_tokens, prev_tokens=prompt_tokens)
                finish_reason = "stop"
                break

            completion_tokens.append(token)

            all_text = self.detokenize(completion_tokens, prev_tokens=prompt_tokens)

            # Contains multi-byte UTF8
            for k, char in enumerate(all_text[-3:]):
                k = 3 - k
                for num, pattern in [(2, 192), (3, 224), (4, 240)]:
                    # Bitwise AND check
                    if num > k and pattern & char == pattern:
                        multibyte_fix = num - k

            # Stop incomplete bytes from passing
            if multibyte_fix > 0:
                multibyte_fix -= 1
                continue

            any_stop = [s for s in stop_sequences if s in all_text]
            if len(any_stop) > 0:
                first_stop = any_stop[0]
                text = all_text[: all_text.index(first_stop)]
                finish_reason = "stop"
                break

            if stream:
                remaining_tokens = completion_tokens[returned_tokens:]
                remaining_text = self.detokenize(remaining_tokens, prev_tokens=prompt_tokens + completion_tokens[:returned_tokens])
                remaining_length = len(remaining_text)

                # We want to avoid yielding any characters from
                # the generated text if they are part of a stop
                # sequence.
                first_stop_position = 0
                for s in stop_sequences:
                    for i in range(min(len(s), remaining_length), 0, -1):
                        if remaining_text.endswith(s[:i]):
                            if i > first_stop_position:
                                first_stop_position = i
                            break

                token_end_position = 0

                if logprobs is not None:
                    # not sure how to handle this branch when dealing
                    # with CJK output, so keep it unchanged
                    for token in remaining_tokens:
                        if token == self.token_bos():
                            continue
                        token_end_position += len(self.detokenize([token], prev_tokens=prompt_tokens + completion_tokens[:returned_tokens]))
                        # Check if stop sequence is in the token
                        if token_end_position > (
                            remaining_length - first_stop_position
                        ):
                            break
                        token_str = self.detokenize([token], prev_tokens=prompt_tokens + completion_tokens[:returned_tokens]).decode(
                            "utf-8", errors="ignore"
                        )
                        text_offset = len(prompt) + len(
                            self.detokenize(completion_tokens[:returned_tokens], prev_tokens=prompt_tokens + completion_tokens[:returned_tokens]).decode(
                                "utf-8", errors="ignore"
                            )
                        )
                        token_offset = len(prompt_tokens) + returned_tokens
                        logits = self._scores[token_offset - 1, :]
                        current_logprobs = Llama.logits_to_logprobs(logits).tolist()
                        sorted_logprobs = list(
                            sorted(
                                zip(current_logprobs, range(len(current_logprobs))),
                                reverse=True,
                            )
                        )
                        top_logprob = {
                            self.detokenize([i]).decode(
                                "utf-8", errors="ignore"
                            ): logprob
                            for logprob, i in sorted_logprobs[:logprobs]
                        }
                        top_logprob.update({token_str: current_logprobs[int(token)]})
                        logprobs_or_none = {
                            "tokens": [
                                self.detokenize([token], prev_tokens=prompt_tokens + completion_tokens[:returned_tokens]).decode(
                                    "utf-8", errors="ignore"
                                )
                            ],
                            "text_offset": [text_offset],
                            "token_logprobs": [current_logprobs[int(token)]],
                            "top_logprobs": [top_logprob],
                        }
                        returned_tokens += 1
                        yield {
                            "id": completion_id,
                            "object": "text_completion",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {
                                    "text": self.detokenize([token], prev_tokens=prompt_tokens + completion_tokens[:returned_tokens]).decode(
                                        "utf-8", errors="ignore"
                                    ),
                                    "index": 0,
                                    "logprobs": logprobs_or_none,
                                    "finish_reason": None,
                                }
                            ],
                        }
                else:
                    while len(remaining_tokens) > 0:
                        decode_success = False
                        for i in range(1, len(remaining_tokens) + 1):
                            try:
                                bs = self.detokenize(remaining_tokens[:i], prev_tokens=prompt_tokens + completion_tokens[:returned_tokens])
                                ts = bs.decode("utf-8")
                                decode_success = True
                                break
                            except UnicodeError:
                                pass
                        else:
                            break
                        if not decode_success:
                            # all remaining tokens cannot be decoded to a UTF-8 character
                            break
                        token_end_position += len(bs)
                        if token_end_position > (
                            remaining_length - first_stop_position
                        ):
                            break
                        remaining_tokens = remaining_tokens[i:]
                        returned_tokens += i

                        yield {
                            "id": completion_id,
                            "object": "text_completion",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {
                                    "text": ts,
                                    "index": 0,
                                    "logprobs": None,
                                    "finish_reason": None,
                                }
                            ],
                        }

            if len(completion_tokens) >= max_tokens:
                text = self.detokenize(completion_tokens, prev_tokens=prompt_tokens)
                finish_reason = "length"
                break

        if stopping_criteria is not None and stopping_criteria(
            self._input_ids, self._scores[-1, :]
        ):
            text = self.detokenize(completion_tokens, prev_tokens=prompt_tokens)
            finish_reason = "stop"

        if self.verbose:
            self._ctx.print_timings()

        if stream:
            remaining_tokens = completion_tokens[returned_tokens:]
            all_text = self.detokenize(remaining_tokens, prev_tokens=prompt_tokens + completion_tokens[:returned_tokens])
            any_stop = [s for s in stop_sequences if s in all_text]
            if len(any_stop) > 0:
                end = min(all_text.index(stop) for stop in any_stop)
            else:
                end = len(all_text)

            token_end_position = 0
            for token in remaining_tokens:
                token_end_position += len(self.detokenize([token], prev_tokens=prompt_tokens + completion_tokens[:returned_tokens]))

                logprobs_or_none: Optional[CompletionLogprobs] = None
                if logprobs is not None:
                    if token == self.token_bos():
                        continue
                    token_str = self.detokenize([token]).decode(
                        "utf-8", errors="ignore"
                    )
                    text_offset = len(prompt) + len(
                        self.detokenize(completion_tokens[:returned_tokens], prev_tokens=prompt_tokens + completion_tokens[:returned_tokens])
                    )
                    token_offset = len(prompt_tokens) + returned_tokens - 1
                    logits = self._scores[token_offset, :]
                    current_logprobs = Llama.logits_to_logprobs(logits).tolist()
                    sorted_logprobs = list(
                        sorted(
                            zip(current_logprobs, range(len(current_logprobs))),
                            reverse=True,
                        )
                    )
                    top_logprob = {
                        self.detokenize([i]).decode("utf-8", errors="ignore"): logprob
                        for logprob, i in sorted_logprobs[:logprobs]
                    }
                    top_logprob.update({token_str: current_logprobs[int(token)]})
                    logprobs_or_none = {
                        "tokens": [
                            self.detokenize([token]).decode("utf-8", errors="ignore")
                        ],
                        "text_offset": [text_offset],
                        "token_logprobs": [current_logprobs[int(token)]],
                        "top_logprobs": [top_logprob],
                    }

                if token_end_position >= end:
                    last_text = self.detokenize([token])
                    if token_end_position == end - 1:
                        break
                    returned_tokens += 1
                    yield {
                        "id": completion_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "text": last_text[
                                    : len(last_text) - (token_end_position - end)
                                ].decode("utf-8", errors="ignore"),
                                "index": 0,
                                "logprobs": logprobs_or_none,
                                "finish_reason": None,
                            }
                        ],
                    }
                    break
                returned_tokens += 1
                yield {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "text": self.detokenize([token]).decode(
                                "utf-8", errors="ignore"
                            ),
                            "index": 0,
                            "logprobs": logprobs_or_none,
                            "finish_reason": None,
                        }
                    ],
                }
            yield {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "text": "",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": finish_reason,
                    }
                ],
            }
            if self.cache:
                if self.verbose:
                    print("Llama._create_completion: cache save", file=sys.stderr)
                self.cache[prompt_tokens + completion_tokens] = self.save_state()
                print("Llama._create_completion: cache saved", file=sys.stderr)
            return

        if self.cache:
            if self.verbose:
                print("Llama._create_completion: cache save", file=sys.stderr)
            self.cache[prompt_tokens + completion_tokens] = self.save_state()

        text_str = text.decode("utf-8", errors="ignore")

        if echo:
            text_str = prompt + text_str

        if suffix is not None:
            text_str = text_str + suffix

        logprobs_or_none: Optional[CompletionLogprobs] = None
        if logprobs is not None:
            text_offset = 0 if echo else len(prompt)
            token_offset = 0 if echo else len(prompt_tokens[1:])
            text_offsets: List[int] = []
            token_logprobs: List[Optional[float]] = []
            tokens: List[str] = []
            top_logprobs: List[Optional[Dict[str, float]]] = []

            if echo:
                # Remove leading BOS token
                all_tokens = prompt_tokens[1:] + completion_tokens
            else:
                all_tokens = completion_tokens

            all_token_strs = [
                self.detokenize([token], prev_tokens=all_tokens[:i]).decode("utf-8", errors="ignore")
                for i, token in enumerate(all_tokens)
            ]
            all_logprobs = Llama.logits_to_logprobs(self._scores)[token_offset:]
            sorted_all_logprobs, sorted_idx = topk_numpy(all_logprobs, logprobs, 1)
            # TODO: may be able to change this loop to use np.take_along_dim
            for idx, (token, token_str, logprobs_token, sorted_logprobs_token) in enumerate(
                zip(all_tokens, all_token_strs, all_logprobs, sorted_all_logprobs)
            ):
                if token == self.token_bos():
                    continue
                text_offsets.append(
                    text_offset
                    + len(
                        self.detokenize(all_tokens[:idx]).decode(
                            "utf-8", errors="ignore"
                        )
                    )
                )
                tokens.append(token_str)
                token_logprobs.append(logprobs_token[int(token)])


                top_logprob: Optional[Dict[str, float]] = {
                    self.detokenize([i], prev_tokens=all_tokens[:idx]).decode("utf-8", errors="ignore"): logprob
                    for logprob, i in zip(sorted_logprobs_token, sorted_idx[idx])
                }

                top_logprob.update({token_str: logprobs_token[int(token)]})
                top_logprobs.append(top_logprob)


            # Weird idosincracy of the OpenAI API where
            # token_logprobs and top_logprobs are null for
            # the first token.
            if echo and len(all_tokens) > 0:
                token_logprobs[0] = None
                top_logprobs[0] = None
            logprobs_or_none = {
                "tokens": tokens,
                "text_offset": text_offsets,
                "token_logprobs": token_logprobs,
                "top_logprobs": top_logprobs,
            }

        yield {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "text": text_str,
                    "index": 0,
                    "logprobs": logprobs_or_none,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": len(completion_tokens),
                "total_tokens": len(prompt_tokens) + len(completion_tokens),
            },
        }



def get_result(logprobs, context_length):
    is_greedy = True
    offsets = logprobs["text_offset"]
    tokens = logprobs["tokens"]
    tokens_logprobs = logprobs["token_logprobs"]

    idx = 0
    while offsets[idx] < context_length:
        idx += 1
    continuation_logprobs = sum(tokens_logprobs[idx:-1])
    for i in range(idx, len(tokens)):
        token = tokens[i]
        top_tokens = logprobs["top_logprobs"][i]
        top_token = max(top_tokens.keys(), key=lambda x: top_tokens[x])
        if top_token != token:
            is_greedy = False
            break

    return continuation_logprobs, is_greedy


@register_model("gguf-custom", "ggml-custom")
class WrapperGGUFLM(LM):
    def __init__(self, pretrained=None, ftype="*q4_0.gguf", max_length=2048, **kwargs):
        super().__init__()
        assert pretrained, "must pass `model` to use GGUF LM!"

        if kwargs["device"] == "cuda":
            n_gpu_layers = kwargs.get("n_gpu_layers", -1)
            split_mode = kwargs.get("split_mode", 1)
            main_gpu = kwargs.get("main_gpu", 0)
        elif kwargs["device"] == "cpu":
            n_gpu_layers = 0
            split_mode = 1
            main_gpu = 0
        else:
            assert False, f"{kwargs['device']} device not support now!"

        self.model = CustomLlamaCpp.from_pretrained(
                repo_id=pretrained,
                filename=ftype,
                n_gpu_layers=n_gpu_layers,
                split_mode=split_mode,
                main_gpu=main_gpu,
                logits_all=True,
                n_ctx=max_length,
                use_mlock=True,
                last_n_tokens_size=64,
                cache=False,
                cache_type='ram',
                verbose=True,)


        if "qwen" in self.model.metadata["general.architecture"]:
            self.model.detokenize = self.model.detokenize_bpe


        self.logprobs = 10
        self.temperature = 0.0
        self.max_length = max_length

    def gguf_completion(
        self, context, continuation=None, stop=None, retries=3, delay=5, **kwargs
    ):
        for _ in range(retries):
            try:
                prompt = context
                request = {
                    "prompt": prompt,
                    "logprobs": self.logprobs,
                    "temperature": self.temperature,
                }
                if continuation:
                    prompt += continuation
                    request.update({"prompt": prompt, "max_tokens": 1, "echo": True})
                if stop is not None:
                    request["stop"] = stop
                response = self.model(**request)
                return response
            except RequestException as e:
                logger.error(f"RequestException: {e}")
                print(request)
                time.sleep(delay)  # wait before retrying
        else:
            raise Exception(f"Failed to get a valid response after {retries} retries.")

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        if not requests:
            return []
        res = []

        for context, continuation in tqdm(
            [req.args for req in requests], disable=disable_tqdm
        ):
            response = self.gguf_completion(context=context, continuation=continuation)
            if response and "choices" in response and response["choices"]:
                choice = response["choices"][0]
                logprobs = choice.get("logprobs")
                if (
                    logprobs
                    and "token_logprobs" in logprobs
                    and logprobs["token_logprobs"]
                ):
                    try:
                        logprob, is_greedy = get_result(logprobs, len(context))
                    except:
                        print(response)
                    res.append((logprob, is_greedy))
                else:
                    logger.warning(
                        "Invalid logprobs data. Expected 'logprobs' to contain 'token_logprobs' list."
                    )
            else:
                logger.error(
                    f"Invalid response for loglikelihood. Response: {response}"
                )
                assert False
        return res

    def generate_until(self, requests, disable_tqdm: bool = False):
        if not requests:
            return []

        res = []
        for request in tqdm([req.args for req in requests], disable=disable_tqdm):
            inp = request[0]
            request_args = request[1]
            until = request_args.get("until", ["</s>"])
            response = self.gguf_completion(context=inp, stop=until)
            if response and "choices" in response and response["choices"]:
                choice = response["choices"][0]
                if "text" in choice:
                    generated_text = choice["text"].strip()
                    res.append(generated_text)
                else:
                    logger.error(
                        f"Invalid response for greedy_until. Response: {response}"
                    )
                    res.append(None)  # Add default value in case of error
            else:
                logger.error(f"Invalid response for greedy_until. Response: {response}")
                res.append(None)  # Add default value in case of error
        return res

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError(
            "loglikelihood_rolling not yet supported for GGUF models"
        )
