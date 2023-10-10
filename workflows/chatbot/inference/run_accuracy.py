import os
import argparse
import json
import re
import gc
import torch
from pathlib import Path
import intel_extension_for_pytorch as ipex

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    T5ForConditionalGeneration,
    AutoTokenizer,
    LlamaTokenizer,
)


MODEL_CLASSES = {
    "gpt-j": (AutoModelForCausalLM, AutoTokenizer),
    "gpt-neox": (AutoModelForCausalLM, AutoTokenizer),
    "opt": (AutoModelForCausalLM, AutoTokenizer),
    "bloom": (AutoModelForCausalLM, AutoTokenizer),
    "llama": (AutoModelForCausalLM, LlamaTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", nargs="?", default="EleutherAI/gpt-j-6b")
parser.add_argument("--output_dir", nargs="?", default="./saved_results")
parser.add_argument("--device", default="cpu", type=str, help="cpu")
parser.add_argument(
    "--dtype", default="bfloat16", type=str, help="float32 or bfloat16 or int8"
)
parser.add_argument("--accuracy-only", action="store_true")
parser.add_argument(
    "--batch-size", default=1, type=int, help="For accuracy measurement only."
)
parser.add_argument(
    "--save-accuracy-path", default=None, help="Save accuracy results path."
)
parser.add_argument(
    "--ipex", action="store_true", help="use intel extension for pytorch."
)
parser.add_argument(
    "--jit", action="store_true", help="convert model to torchscript mode."
)
parser.add_argument("--int8-bf16-mixed", action="store_true", help="int8 mixed bf16")
parser.add_argument("--quantized-model-path", default="./saved_result/best_model.pt")
parser.add_argument(
    "--tasks",
    nargs="+",
    default=[
        "lambada_openai",
    ],
    type=str,
    help="tasks list for accuracy validation, only enabled lambada_openai and lambada_standard at present",
)


args = parser.parse_args()


if args.accuracy_only:
    import lm_eval
    from lm_eval import tasks, evaluator, models
    from lm_eval.base import BaseLM
    from typing import Union, List, Optional
    from transformers import BatchEncoding

    TokenSequence = Union[List[int], torch.LongTensor, torch.Tensor, BatchEncoding]

    class HuggingFaceModel(BaseLM):
        _DEFAULT_MAX_LENGTH = 2048

        def __init__(
            self,
            device="cpu",
            model_id="",
            with_ipex=True,
            with_jit=True,
            with_greedy=False,
            batch_size=1,
            max_length=None,
            dtype: Optional[Union[str, torch.dtype]] = "auto",
        ):
            super().__init__()

            self._device = device
            self._batch_size = batch_size
            self._with_jit = with_jit
            self._with_ipex = with_ipex
            self._with_greedy = with_greedy
            self._max_length = max_length
            self._dtype = dtype

            if dtype == "float16":
                load_dtype = torch.half
                infer_dtype = torch.half
            elif dtype == "bfloat16":
                load_dtype = torch.bfloat16
                infer_dtype = torch.bfloat16
            elif dtype == "int8":
                load_dtype = torch.float32
                infer_dtype = torch.int8
            elif dtype == "float32":
                load_dtype = torch.float32
                infer_dtype = torch.float32

            amp_enabled = True if dtype != "float32" else False
            amp_dtype = getattr(torch, dtype)

            model_type = next(
                (x for x in MODEL_CLASSES.keys() if x in model_id.lower()), "auto"
            )
            model_class = MODEL_CLASSES[model_type]
            self.tokenizer = model_class[1].from_pretrained(model_id)

            self.config = AutoConfig.from_pretrained(model_id, torchscript=with_jit)

            self.model = model_class[0].from_pretrained(
                model_id,
                low_cpu_mem_usage=True,
                config=self.config,
                torch_dtype=load_dtype,
            )

            self.model = self.model.eval()

            if with_ipex:
                self.model = ipex._optimize_transformers(
                    self.model.eval(), dtype=infer_dtype, inplace=True
                )

            self.base_model = self.model

            self.iter = 0
            self.num_beams = 1 if with_greedy else 4
            self.tp_number = 1

        def _model_call(
            self, inputs: TokenSequence, labels: Optional[TokenSequence] = None
        ) -> TokenSequence:
            _attention_mask = []
            _position_ids = []


            if self._with_jit:
                for text in inputs:
                    input_ids = text.to(self._device)
                    input_bs = inputs.shape[0] * self.num_beams
                    if re.search("GPTJ", self.base_model.config.architectures[0]):
                        beam_idx_tmp = torch.zeros(
                            (2048, int(input_bs)), dtype=torch.long
                        ).contiguous()
                        past_key_values = tuple(
                            [
                                (
                                    torch.zeros(
                                        [
                                            1,
                                            int(
                                                self.base_model.config.n_head
                                                / self.tp_number
                                            ),
                                            1,
                                            int(
                                                self.base_model.config.n_embd
                                                / self.base_model.config.n_head
                                            ),
                                        ]
                                    ).contiguous(),
                                    torch.zeros(
                                        [
                                            1,
                                            int(
                                                self.base_model.config.n_head
                                                / self.tp_number
                                            ),
                                            1,
                                            int(
                                                self.base_model.config.n_embd
                                                / self.base_model.config.n_head
                                            ),
                                        ]
                                    ).contiguous(),
                                    beam_idx_tmp,
                                    torch.zeros(1, dtype=torch.long).contiguous(),
                                )
                                for i in range(self.base_model.config.n_layer)
                            ]
                        )
                    elif re.search(
                        "llama", self.base_model.config.architectures[0], re.IGNORECASE
                    ):
                        beam_idx_tmp = torch.zeros(
                            (2048, int(input_bs)), dtype=torch.long
                        ).contiguous()
                        past_key_values = tuple(
                            [
                                (
                                    torch.zeros(
                                        [
                                            1,
                                            int(
                                                self.base_model.config.num_attention_heads
                                                / self.tp_number
                                            ),
                                            1,
                                            int(
                                                self.base_model.config.hidden_size
                                                / self.base_model.config.num_attention_heads
                                            ),
                                        ]
                                    ).contiguous(),
                                    torch.zeros(
                                        [
                                            1,
                                            int(
                                                self.base_model.config.num_attention_heads
                                                / self.tp_number
                                            ),
                                            1,
                                            int(
                                                self.base_model.config.hidden_size
                                                / self.base_model.config.num_attention_heads
                                            ),
                                        ]
                                    ).contiguous(),
                                    beam_idx_tmp,
                                    torch.zeros(1, dtype=torch.long).contiguous(),
                                )
                                for i in range(self.base_model.config.num_hidden_layers)
                            ]
                        )
                    elif re.search(
                        "gptneox", self.base_model.config.architectures[0], re.IGNORECASE
                    ):
                        beam_idx_tmp = torch.zeros(
                            (2048, int(input_bs)), dtype=torch.long
                        ).contiguous()
                        past_key_values = tuple(
                            [
                                (
                                    torch.zeros(
                                        [
                                            1,
                                            int(
                                                self.base_model.config.num_attention_heads
                                                / self.tp_number
                                            ),
                                            1,
                                            int(
                                                self.base_model.config.hidden_size
                                                / self.base_model.config.num_attention_heads
                                            ),
                                        ]
                                    ).contiguous(),
                                    torch.zeros(
                                        [
                                            1,
                                            int(
                                                self.base_model.config.num_attention_heads
                                                / self.tp_number
                                            ),
                                            1,
                                            int(
                                                self.base_model.config.hidden_size
                                                / self.base_model.config.num_attention_heads
                                            ),
                                        ]
                                    ).contiguous(),
                                    beam_idx_tmp,
                                    torch.zeros(1, dtype=torch.long).contiguous(),
                                )
                                for i in range(self.base_model.config.num_hidden_layers)
                            ]
                        )
                    elif re.search(
                        "bloom", self.base_model.config.architectures[0], re.IGNORECASE
                    ):
                        past_key_values = tuple(
                            [
                                (
                                    torch.zeros(
                                        [
                                            1,
                                            int(self.base_model.config.n_head),
                                            1,
                                            int(
                                                self.base_model.config.hidden_size
                                                / self.base_model.config.n_head
                                            ),
                                        ]
                                    ),
                                    torch.zeros(
                                        [
                                            1,
                                            int(self.base_model.config.n_head),
                                            1,
                                            int(
                                                self.base_model.config.hidden_size
                                                / self.base_model.config.n_head
                                            ),
                                        ]
                                    ),
                                )
                                for i in range(self.base_model.config.n_layer)
                            ]
                        )
                    elif re.search(
                        "OPT", self.base_model.config.architectures[0], re.IGNORECASE
                    ):
                        beam_idx_tmp = torch.zeros(
                            (2048, int(input_bs)), dtype=torch.long
                        ).contiguous()
                        past_key_values = tuple(
                            [
                                (
                                    torch.zeros(
                                        [
                                            1,
                                            int(
                                                self.base_model.config.num_attention_heads
                                                / self.tp_number
                                            ),
                                            1,
                                            int(
                                                self.base_model.config.hidden_size
                                                / self.base_model.config.num_attention_heads
                                            ),
                                        ]
                                    ).contiguous(),
                                    torch.zeros(
                                        [
                                            1,
                                            int(
                                                self.base_model.config.num_attention_heads
                                                / self.tp_number
                                            ),
                                            1,
                                            int(
                                                self.base_model.config.hidden_size
                                                / self.base_model.config.num_attention_heads
                                            ),
                                        ]
                                    ).contiguous(),
                                    beam_idx_tmp,
                                    torch.zeros(1, dtype=torch.long).contiguous(),
                                )
                                for i in range(self.base_model.config.num_hidden_layers)
                            ]
                        )

                    position_ids = torch.arange(len(input_ids))
                    attention_mask = torch.ones(len(input_ids))

                    _attention_mask.append(attention_mask)
                    _position_ids.append(position_ids)

                attention_mask_batched = torch.stack(_attention_mask)
                position_ids_batched = torch.stack(_position_ids)

            if self._with_jit and self.iter == 0 and self._dtype == "int8":
                with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
                    enabled=True
                    if args.int8_bf16_mixed or self._dtype == torch.bfloat16
                    else False,
                    dtype=torch.bfloat16,
                ):
                    if self._dtype != "int8":
                        if re.search(
                            "bloom",
                            self.base_model.config.architectures[0],
                            re.IGNORECASE,
                        ) or re.search(
                            "OPT",
                            self.base_model.config.architectures[0],
                            re.IGNORECASE,
                        ):
                            example_dict = {
                                "input_ids": inputs,
                                "attention_mask": attention_mask_batched,
                                "past_key_values": past_key_values,
                            }
                        else:
                            example_dict = {
                                "input_ids": inputs,
                                "attention_mask": attention_mask_batched,
                                "position_ids": position_ids_batched,
                                "past_key_values": past_key_values,
                            }

                            self.model = torch.jit.trace(
                                self.model.eval(),
                                example_kwarg_inputs=example_dict,
                                strict=False,
                                check_trace=False,
                            )
                            self.model = torch.jit.freeze(self.model.eval())
                    else:
                        self.model = torch.jit.load(args.quantized_model_path)
                        self.model = torch.jit.freeze(self.model.eval())

                    if re.search(
                        "bloom", self.base_model.config.architectures[0], re.IGNORECASE
                    ) or re.search(
                        "OPT", self.base_model.config.architectures[0], re.IGNORECASE
                    ):
                        self.model(
                            inputs,
                            past_key_values=past_key_values,
                            attention_mask=attention_mask_batched,
                        )
                        self.model(
                            inputs,
                            past_key_values=past_key_values,
                            attention_mask=attention_mask_batched,
                        )
                    else:
                        self.model(
                            inputs,
                            past_key_values=past_key_values,
                            attention_mask=attention_mask_batched,
                            position_ids=position_ids_batched,
                        )
                        self.model(
                            inputs,
                            past_key_values=past_key_values,
                            attention_mask=attention_mask_batched,
                            position_ids=position_ids_batched,
                        )

                self.iter = self.iter + 1

            if re.search(
                "bloom", self.base_model.config.architectures[0], re.IGNORECASE
            ) or re.search(
                "OPT", self.base_model.config.architectures[0], re.IGNORECASE
            ):
                with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
                    enabled=True
                    if args.int8_bf16_mixed or self._dtype == torch.bfloat16
                    else False,
                    dtype=torch.bfloat16,
                ):
                    if self._with_jit:
                        output = self.model(
                            inputs,
                            past_key_values=past_key_values,
                            attention_mask=attention_mask_batched,
                        )
                    else:
                        output = self.base_model(
                            inputs,
                        )
            else:
                with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
                    enabled=True
                    if args.int8_bf16_mixed or self._dtype == torch.bfloat16
                    else False,
                    dtype=torch.bfloat16,
                ):
                    if self._with_jit:
                        output = self.model(
                            inputs,
                            past_key_values=past_key_values,
                            attention_mask=attention_mask_batched,
                            position_ids=position_ids_batched,
                        )
                    else:
                        output = self.base_model(
                            inputs,
                        )

            if isinstance(output, tuple):
                return output[0]

            return output["logits"]

        @property
        def eot_token_id(self):
            # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
            return self.tokenizer.eos_token_id

        @property
        def max_length(self):
            if self._max_length:  # if max length manually set, return it
                return self._max_length
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
        def max_gen_toks(self):
            return 256

        @property
        def batch_size(self):
            # TODO: fix multi-gpu
            return self._batch_size  # * gpus

        @property
        def device(self):
            # TODO: fix multi-gpu
            return self._device

        def tok_encode(self, string: str):
            return self.tokenizer.encode(string, add_special_tokens=False)

        def tok_decode(self, tokens):
            return self.tokenizer.decode(tokens)

        def _model_generate(self, context, max_length, eos_token_id):
            generation_kwargs = {"do_sample": False, "max_length": max_length}
            if eos_token_id is not None:
                generation_kwargs["eos_token_id"] = eos_token_id
                generation_kwargs[
                    "pad_token_id"
                ] = eos_token_id  # setting eos_token_id as pad token
            return self.model.generate(context, **generation_kwargs)

    task_dict = lm_eval.tasks.get_task_dict(args.tasks)

    hfmodel = HuggingFaceModel(
        model_id=args.model,
        device="cpu",
        batch_size=args.batch_size,
        with_ipex=args.ipex,
        with_jit=args.jit,
        dtype=args.dtype,
    )

    results = evaluator.evaluate(
        hfmodel,
        task_dict,
        #        bootstrap_iters=1000,
        #        limit=100
    )

    print(evaluator.make_table(results))
