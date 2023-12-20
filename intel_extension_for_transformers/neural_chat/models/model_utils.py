#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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

import json
from pathlib import Path
import copy, time
from datetime import datetime
import sys
import torch
import transformers
import warnings
from queue import Queue
import re, os
from threading import Thread
import contextlib
from huggingface_hub import snapshot_download
import logging
logging.basicConfig(
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    datefmt="%d-%M-%Y %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)
from typing import List
from transformers import (
    GenerationConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoConfig,
    TextIteratorStreamer,
    StoppingCriteriaList,
    StoppingCriteria
)
from transformers.deepspeed import is_deepspeed_available
from transformers.utils import is_bitsandbytes_available, is_offline_mode
from intel_extension_for_transformers.transformers import (
    MixedPrecisionConfig,
    WeightOnlyQuantConfig,
    BitsAndBytesConfig
)

import shutil

if is_deepspeed_available():
    import deepspeed # pylint: disable=E0401

class StopOnTokens(StoppingCriteria):
    def __init__(self, min_length: int, start_length: int, stop_token_id: List[int]):
        self.min_length = min_length
        self.start_length = start_length
        self.stop_token_id = stop_token_id

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        if scores is not None:
            if len(scores) > self.min_length:
                for stop_id in self.stop_token_id:
                    if input_ids[0][self.start_length - 1 + len(scores)] == stop_id:
                        return True
        elif input_ids.shape[-1] - self.start_length > self.min_length:
            for stop_id in self.stop_token_id:
                if input_ids[0][input_ids.shape[-1] - 1] == stop_id:
                    return True
        return False

def get_repo_root(model_name_or_path, local_rank=-1, token=None):
    """
    Downloads the specified model checkpoint and returns the repository where it was downloaded.
    """
    if Path(model_name_or_path).is_dir():
        # If it is a local model, no need to download anything
        return model_name_or_path
    else:
        # Checks if online or not
        if is_offline_mode():
            if local_rank == 0:
                logging.info("Offline mode: forcing local_files_only=True")

        # Only download PyTorch weights by default
        allow_patterns = ["*.bin"]

        # Download only on first process
        if local_rank in [-1, 0]:
            cache_dir = snapshot_download(
                model_name_or_path,
                local_files_only=is_offline_mode(),
                cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
                allow_patterns=allow_patterns,
                max_workers=16,
                token=token,
            )
            if local_rank == -1:
                # If there is only one process, then the method is finished
                return cache_dir

        # Make all processes wait so that other processes can get the checkpoint directly from cache
        torch.distributed.barrier()

        return snapshot_download(
            model_name_or_path,
            local_files_only=is_offline_mode(),
            cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
            allow_patterns=allow_patterns,
            token=token,
        )


def get_checkpoint_files(model_name_or_path, local_rank, token=None):
    """
    Gets the list of files for the specified model checkpoint.
    """
    cached_repo_dir = get_repo_root(model_name_or_path, local_rank, token)

    # Extensions: .bin | .pt
    # Creates a list of paths from all downloaded files in cache dir
    file_list = [str(entry) for entry in Path(cached_repo_dir).rglob("*.[bp][it][n]") if entry.is_file()]
    return file_list


def write_checkpoints_json(model_name_or_path, local_rank, checkpoints_json, token=None):
    """
    Dumps metadata into a JSON file for DeepSpeed-inference.
    """
    checkpoint_files = get_checkpoint_files(model_name_or_path, local_rank, token)
    if local_rank == 0 and len(checkpoint_files) != 0:
        data = {"type": "ds_model", "checkpoints": checkpoint_files, "version": 1.0}
        with open(checkpoints_json, "w") as fp:
            json.dump(data, fp)
    return len(checkpoint_files) != 0


def model_on_meta(config):
    """
    Checks if load the model to meta.
    """
    return config.model_type in ["bloom", "llama"]


def get_optimized_model_name(config):
    # pylint: disable=E0401
    # pylint: disable=E0611
    from optimum.habana.transformers.generation import MODELS_OPTIMIZED_WITH_STATIC_SHAPES

    for model_type in MODELS_OPTIMIZED_WITH_STATIC_SHAPES:
        if model_type == config.model_type:
            return model_type

    return None


def model_is_optimized(config):
    """
    Checks if the given config belongs to a model in optimum/habana/transformers/models, which has a
    new input token_idx.
    """
    return get_optimized_model_name(config) is not None


def get_ds_injection_policy(config):
    model_type = get_optimized_model_name(config)
    policy = {}
    if model_type:
        if model_type == "bloom":
            from transformers.models.bloom.modeling_bloom import BloomBlock

            policy = {BloomBlock: ("self_attention.dense", "mlp.dense_4h_to_h")}

        if model_type == "opt":
            from transformers.models.opt.modeling_opt import OPTDecoderLayer

            policy = {OPTDecoderLayer: ("self_attn.out_proj", ".fc2")}

        if model_type == "gpt2":
            from transformers.models.gpt2.modeling_gpt2 import GPT2MLP

            policy = {GPT2MLP: ("attn.c_proj", "mlp.c_proj")}

        if model_type == "gptj":
            from transformers.models.gptj.modeling_gptj import GPTJBlock

            policy = {GPTJBlock: ("attn.out_proj", "mlp.fc_out")}

        if model_type == "gpt_neox":
            from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer

            policy = {GPTNeoXLayer: ("attention.dense", "mlp.dense_4h_to_h")}

        if model_type == "llama":
            from transformers.models.llama.modeling_llama import LlamaDecoderLayer

            policy = {LlamaDecoderLayer: ("self_attn.o_proj", "mlp.down_proj")}

    return policy

def max_input_len(input_text_length):
    if input_text_length <= 128:
        return 128
    elif input_text_length <= 512:
        return 512
    elif input_text_length <= 2048:
        return 2048
    else:
        logging.info("Max support length is 4096")
        return 4096


MODELS = {}


def smart_context_manager(use_deepspeed=False, model_dtype=torch.bfloat16):
    if use_deepspeed:
        ctx_manager = deepspeed.OnDevice(dtype=model_dtype, device="cpu")
    else:
        ctx_manager = contextlib.nullcontext()
    return ctx_manager


def import_deepspeed():
    if not is_deepspeed_available():
        raise ImportError(
            "This script requires deepspeed: `pip install"
            " git+https://github.com/HabanaAI/DeepSpeed.git@1.11.0`."
        )
    # Initialize process(es) for DeepSpeed
    deepspeed.init_distributed(dist_backend="hccl")
    logging.info("DeepSpeed is enabled.")


def init_deepspeed_inference(model, model_name_or_path, peft_path, use_hpu_graphs, is_meta, token=None):
    # Initialize the model
    from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu # pylint: disable=E0401

    world_size, rank, local_rank = initialize_distributed_hpu()
    merged_model_dir = None
    if peft_path and is_meta:
        merged_model_dir = "/tmp/text_generation_merged_peft_model"
        if local_rank == 0:
            if Path(merged_model_dir).is_dir():
                shutil.rmtree(merged_model_dir)
            peft_model(model_name_or_path, peft_path, torch.bfloat16, token).save_pretrained(merged_model_dir)
        torch.distributed.barrier()

    model = model.eval()
    ds_inference_kwargs = {"dtype": torch.bfloat16}
    ds_inference_kwargs["tensor_parallel"] = {"tp_size": world_size}
    ds_inference_kwargs["enable_cuda_graph"] = use_hpu_graphs
    # Make sure all devices/nodes have access to the model checkpoints
    if is_meta:
        checkpoints_json = "checkpoints.json"
        ret = write_checkpoints_json(merged_model_dir if merged_model_dir is not None else model_name_or_path,
                local_rank, checkpoints_json, token)
        if ret == False:
            is_meta = False
            generation_config = model.generation_config
            model = AutoModelForCausalLM.from_pretrained(
                merged_model_dir if merged_model_dir is not None else model_name_or_path,
                use_auth_token=token,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            )
            model.generation_config = generation_config


    torch.distributed.barrier()

    if is_meta:
        ds_inference_kwargs["checkpoint"] = checkpoints_json

    ds_inference_kwargs["injection_policy"] = get_ds_injection_policy(model.config)
    model = deepspeed.init_inference(model, **ds_inference_kwargs)
    return model.module


def peft_model(model_name, peft_model, model_dtype, hf_access_token=None):
    import importlib.util

    if importlib.util.find_spec("peft") is None:
        raise ImportError("The `peft` package is not installed, please run: `pip install peft`.")
    from peft import AutoPeftModelForCausalLM
    from peft.config import PeftConfigMixin

    base_model_name = PeftConfigMixin.from_pretrained(
        peft_model,
        use_auth_token=hf_access_token,
    ).base_model_name_or_path

    base_model_is_local = Path(base_model_name).is_dir()
    if not base_model_is_local:
        # Check if the base model path to a remote repository on the HF Hub exists
        from huggingface_hub import list_repo_files

        try:
            list_repo_files(base_model_name)
            base_model_is_remote = True
        except Exception:
            base_model_is_remote = False

    if base_model_is_local or base_model_is_remote:
        model = AutoPeftModelForCausalLM.from_pretrained(peft_model, torch_dtype=model_dtype, low_cpu_mem_usage=True,
                                                         use_auth_token=hf_access_token)
    else:
        # Since the base model doesn't exist locally nor remotely, use `args.model_name_or_path` as the base model
        print(
            f"The base model `{base_model_name}` of the LoRA configuration associated"
            f" to `{peft_model}` does not exist locally or remotely. Using "
            f"`--model_name_or_path {model_name}` as a fall back for the base model."
        )
        from peft import PeftModel

        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=model_dtype, low_cpu_mem_usage=True,
                                                     use_auth_token=hf_access_token)
        model = PeftModel.from_pretrained(model, peft_model, torch_dtype=model_dtype, low_cpu_mem_usage=True,
                                          use_auth_token=hf_access_token)

    return model.merge_and_unload()

def load_model(
    model_name,
    tokenizer_name,
    device="cpu",
    use_hpu_graphs=False,
    cpu_jit=False,
    ipex_int8=False,
    use_cache=True,
    peft_path=None,
    use_deepspeed=False,
    optimization_config=None,
    hf_access_token=None,
    use_llm_runtime=False,
    assistant_model=None
):
    """
    Load the model and initialize the tokenizer.

    Args:
        model_name (str): The name of the model.
        device (str, optional): The device for the model. Defaults to 'cpu'. The valid value is 'cpu', 'cuda' or 'hpu'.
        use_hpu_graphs (bool, optional): Whether to use HPU graphs. Defaults to False. Only set when device is hpu.
        assistant_model (str, optional): The assistant model name. Defaults to None.

    Returns:
        None

    Raises:
        ValueError
    """
    print("Loading model {}".format(model_name))
    if device == "hpu":
        if use_deepspeed:
            import_deepspeed()
        # Tweak generation so that it runs faster on Gaudi
        # pylint: disable=E0401
        # pylint: disable=E0611
        from optimum.habana.transformers.modeling_utils import (
            adapt_transformers_to_gaudi,
        )

        adapt_transformers_to_gaudi()

    if isinstance(optimization_config, MixedPrecisionConfig):
        dtype = optimization_config.dtype
    else:
        dtype = "float32"

    bitsandbytes_quant_config = None
    if isinstance(optimization_config, BitsAndBytesConfig):
        if device == "cuda" and is_bitsandbytes_available() and torch.cuda.is_available():
            bitsandbytes_quant_config = optimization_config
        else:
            logging.warning(
                "CUDA device or bitsandbytes is not available, please make sure CUDA device and bitsandbytes" \
                + " library is available, ignoring bitsandbytes config now."
            )

    if dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "float32":
        torch_dtype = torch.float32
    else:
        logging.warning(f"Unsupported dtype {dtype}, using float32 now.")
        torch_dtype = torch.float32

    MODELS[model_name] = {}

    # load assistant model
    if assistant_model:
        print("Loading assistant model...")
        assistant_model_class = AutoModelForCausalLM
        print(f"Loading assistant model via {assistant_model_class}")
        assis_model = assistant_model_class.from_pretrained(
            assistant_model,
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype)
        assis_model = assis_model.eval().to(device)
        assis_model = assis_model.to(memory_format=torch.channels_last)
        MODELS[model_name]["assistant_model"] = assis_model
    else:
        MODELS[model_name]["assistant_model"] = None

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=False if (re.search("llama", model_name, re.IGNORECASE)
                or re.search("neural-chat-7b-v2", model_name, re.IGNORECASE)) else True,
            use_auth_token=hf_access_token,
            trust_remote_code=True if (re.search("qwen", model_name, re.IGNORECASE) or \
                re.search("chatglm", model_name, re.IGNORECASE)) else False,
        )
    except EnvironmentError as e:
        if "not a local folder and is not a valid model identifier" in str(e):
            raise ValueError("load_model: tokenizer is not found")
        else:
            raise

    try:
        config = AutoConfig.from_pretrained(model_name, use_auth_token=hf_access_token, trust_remote_code=True \
                                            if (re.search("chatglm", model_name, re.IGNORECASE) or \
                                               re.search("qwen", model_name, re.IGNORECASE)) else False)
    except ValueError as e:
        if "Unrecognized model in" in str(e):
            raise ValueError("load_model: model config is not found")
        else:
            raise

    load_to_meta = model_on_meta(config)

    if isinstance(optimization_config, WeightOnlyQuantConfig):
        from intel_extension_for_transformers.neural_chat.chatbot import optimize_model
        model = optimize_model(model_name, optimization_config, use_llm_runtime)
        if not model.config.is_encoder_decoder:
            tokenizer.padding_side = "left"
        if tokenizer.pad_token is None and tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        MODELS[model_name]["model"] = model
        MODELS[model_name]["tokenizer"] = tokenizer
        logging.info("Optimized Model loaded.")
        return

    try:
        if device == "hpu" and use_deepspeed and load_to_meta:
            with deepspeed.OnDevice(dtype=torch.bfloat16, device="meta"):
                model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
        elif re.search("flan-t5", model_name, re.IGNORECASE) and not ipex_int8:
            with smart_context_manager(use_deepspeed=use_deepspeed):
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    use_auth_token=hf_access_token,
                    quantization_config=bitsandbytes_quant_config,
                )
        elif re.search("chatglm", model_name, re.IGNORECASE) and not ipex_int8:
            with smart_context_manager(use_deepspeed=use_deepspeed):
                model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    use_auth_token=hf_access_token,
                    trust_remote_code=True)
        elif ((
            re.search("gpt", model_name, re.IGNORECASE)
            or re.search("mpt", model_name, re.IGNORECASE)
            or re.search("bloom", model_name, re.IGNORECASE)
            or re.search("llama", model_name, re.IGNORECASE)
            or re.search("magicoder", model_name, re.IGNORECASE)
            or re.search("neural-chat-7b-v1", model_name, re.IGNORECASE)
            or re.search("neural-chat-7b-v2", model_name, re.IGNORECASE)
            or re.search("neural-chat-7b-v3", model_name, re.IGNORECASE)
            or re.search("qwen", model_name, re.IGNORECASE)
            or re.search("starcoder", model_name, re.IGNORECASE)
            or re.search("codellama", model_name, re.IGNORECASE)
            or re.search("mistral", model_name, re.IGNORECASE)
            or re.search("codegen", model_name, re.IGNORECASE)
        ) and not ipex_int8) or re.search("opt", model_name, re.IGNORECASE):
            with smart_context_manager(use_deepspeed=use_deepspeed):
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    use_auth_token=hf_access_token,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    quantization_config=bitsandbytes_quant_config,
                    trust_remote_code=True if (re.search("qwen", model_name, re.IGNORECASE) or \
                        re.search("codegen", model_name, re.IGNORECASE)) else False
                )
        elif (
                (re.search("starcoder", model_name, re.IGNORECASE)
                or re.search("codellama", model_name, re.IGNORECASE)
                or re.search("codegen", model_name, re.IGNORECASE)
                ) and ipex_int8
            ):
            with smart_context_manager(use_deepspeed=use_deepspeed):
                try:
                    import intel_extension_for_pytorch as ipex
                except ImportError:
                    warnings.warn(
                        "Please install Intel Extension for PyTorch to accelerate the model inference."
                    )
                assert ipex.__version__ >= "2.1.0+cpu", "Please use Intel Extension for PyTorch >=2.1.0+cpu."
                from optimum.intel.generation.modeling import TSModelForCausalLM
                model = TSModelForCausalLM.from_pretrained(
                        model_name,
                        file_name="best_model.pt",
                    )
        elif(
                (re.search("llama", model_name, re.IGNORECASE)
                or re.search("opt", model_name, re.IGNORECASE)
                or re.search("gpt_neox", model_name, re.IGNORECASE)
                or re.search("gptj", model_name, re.IGNORECASE)
                or re.search("falcon", model_name, re.IGNORECASE)
                ) and ipex_int8
        ):
            with smart_context_manager(use_deepspeed=use_deepspeed):
                try:
                    import intel_extension_for_pytorch as ipex
                except ImportError:
                    warnings.warn(
                        "Please install Intel Extension for PyTorch to accelerate the model inference."
                    )
                assert ipex.__version__ >= "2.1.0+cpu", "Please use Intel Extension for PyTorch >=2.1.0+cpu."
                if re.search("falcon", model_name, re.IGNORECASE):
                    assert transformers.__version__ <= "4.33.3", "Please pip install transformers==4.33.3"
                from intel_extension_for_transformers.llm.evaluation.models import TSModelCausalLMForITREX
                model = TSModelCausalLMForITREX.from_pretrained(
                    model_name,
                    file_name="best_model.pt"
                )
        else:
            raise ValueError(f"unsupported model name or path {model_name}, \
            only supports FLAN-T5/LLAMA/MPT/GPT/BLOOM/OPT/QWEN/NEURAL-CHAT/MISTRAL/CODELLAMA/STARCODER/CODEGEN now.")
    except EnvironmentError as e:
        if "not a local folder and is not a valid model identifier" in str(e):
            raise ValueError("load_model: model name or path is not found")
        else:
            raise

    if re.search("llama", model.config.architectures[0], re.IGNORECASE):
        # unwind broken decapoda-research config
        model.generation_config.pad_token_id = 0
        model.generation_config.bos_token_id = 1
        model.generation_config.eos_token_id = 2

    if (
        hasattr(model.generation_config, "pad_token_id")
        and model.generation_config.pad_token_id is not None
        and not "chatglm" in model_name
    ):
        tokenizer.pad_token_id = model.generation_config.pad_token_id
    if (
        hasattr(model.generation_config, "eos_token_id")
        and model.generation_config.eos_token_id is not None
        and not "chatglm" in model_name
    ):
        tokenizer.eos_token_id = model.generation_config.eos_token_id
    if (
        hasattr(model.generation_config, "bos_token_id")
        and model.generation_config.bos_token_id is not None
    ):
        tokenizer.bos_token_id = model.generation_config.bos_token_id

    if tokenizer.pad_token_id is None:
        model.generation_config.pad_token_id = (
            tokenizer.pad_token_id
        ) = tokenizer.eos_token_id

    if model.generation_config.eos_token_id is None:
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    if device == "hpu":
        if peft_path and not (use_deepspeed and load_to_meta):
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, peft_path)
            model = model.to(torch.bfloat16)
            model = model.merge_and_unload()

        if not use_deepspeed:
            model = model.eval().to("hpu")
            if use_hpu_graphs:
                from habana_frameworks.torch.hpu import wrap_in_hpu_graph # pylint: disable=E0401
                model = wrap_in_hpu_graph(model)

        if use_deepspeed:
            model = init_deepspeed_inference(
                model=model,
                model_name_or_path=model_name,
                peft_path=peft_path,
                use_hpu_graphs=use_hpu_graphs,
                is_meta=load_to_meta,
                token=hf_access_token,
            )
    else:
        if peft_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, peft_path)
            model = model.to(dtype=torch_dtype)

        if device == "cpu":
            if torch_dtype == torch.bfloat16 and not ipex_int8:
                import intel_extension_for_pytorch as intel_ipex

                model = intel_ipex.optimize(
                    model.eval(),
                    dtype=torch_dtype,
                    inplace=True,
                    level="O1",
                    auto_kernel_selection=True,
                )
                if cpu_jit and (re.search("mpt-7b", model_name, re.IGNORECASE)
                                or re.search("neural-chat-7b-v1", model_name, re.IGNORECASE)):
                    from intel_extension_for_transformers.llm.utils.mpt_trace import \
                        jit_trace_mpt_7b, MPTTSModelForCausalLM

                    model.config.use_cache = use_cache
                    model = jit_trace_mpt_7b(model)
                    config = AutoConfig.from_pretrained(
                        model_name, use_auth_token=hf_access_token
                    )
                    model = MPTTSModelForCausalLM(
                        model, config, use_cache=use_cache, model_dtype=torch.bfloat16
                    )
        elif device in ["cuda", "xpu"]:
            if hasattr(model, "device") and model.device.type != device:
                model = model.eval().to(device)
        else:
            raise ValueError(
                f"unsupported device {device}, only supports cpu, xpu, cuda and hpu now."
            )

    if not model.config.is_encoder_decoder:
        tokenizer.padding_side = "left"

    if tokenizer.pad_token is None and tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
    # warmup for int8 model
    if ipex_int8:
        input_ids = tokenizer("A chat between a curious human and an artificial intelligence assistant.\n"
                              " Human: Tell me about Intel.\n Assistant:", return_tensors="pt").input_ids.to('cpu')
        with torch.inference_mode(), torch.no_grad():
            for i in range(2):
                model.generate(input_ids, max_new_tokens=32, do_sample=False, temperature=0.9)
    MODELS[model_name]["model"] = model
    MODELS[model_name]["tokenizer"] = tokenizer
    print("Model loaded.")

def prepare_inputs(inputs, device):
    return {k:v.to(device=device) for k,v in inputs.items() if torch.is_tensor(v)}

def get_stop_token_ids(model, tokenizer):
    if hasattr(model, 'generation_config') and hasattr(model.generation_config, 'eos_token_id'):
        eos_token_id = model.generation_config.eos_token_id
    else:
        eos_token_id = tokenizer.eos_token_id

    if isinstance(eos_token_id, list):
        stop_token_ids = copy.deepcopy(eos_token_id)
    else:
        stop_token_ids = [eos_token_id]
    end_token_id = torch.flatten(tokenizer("go.", return_tensors="pt").input_ids)[-1]
    stop_token_ids.append(end_token_id)
    return stop_token_ids

def tokenization(prompt, tokenizer, device):
    if device == "hpu":
        input_tokens_no_pad = tokenizer([prompt], return_tensors="pt")
        input_token_len = input_tokens_no_pad.input_ids.shape[-1]
        input_tokens = tokenizer.batch_encode_plus(
            [prompt],
            return_tensors="pt",
            padding="max_length",
            max_length=max_input_len(input_token_len),
        )
    else:
        input_tokens = tokenizer.batch_encode_plus(
            [prompt], return_tensors="pt", padding=True
        )
        input_token_len = input_tokens.input_ids.shape[-1]
    return input_tokens, input_token_len

def get_generate_kwargs(
        max_new_tokens, input_token_len, stop_token_id, assistant_model=None):
    generate_kwargs = {
        "stopping_criteria": StoppingCriteriaList(
            [
                StopOnTokens(
                    min_length=max(max_new_tokens - 20, 0),
                    start_length=input_token_len,
                    stop_token_id=stop_token_id,
                )
            ]
        )
    }
    if assistant_model:
        generate_kwargs["assistant_model"] = assistant_model
        generate_kwargs["use_cache"] = True
    return generate_kwargs

def is_llm_runtime_model(model):
    from intel_extension_for_transformers.llm.runtime.graph import Model
    if isinstance(model, Model):
        return True
    else:
        return False

def remove_prompt_history(model_name, prompt):
    result = prompt
    if re.search("llama", model_name, re.IGNORECASE):
        matches = re.findall(r'\[INST\](.*?)\[/INST\]', prompt)
        if matches:
            result = "[INST]" + matches[-1] + "[/INST]"
    elif re.search("chatglm", model_name, re.IGNORECASE):
        pattern = re.compile(r'问：.*?\n答：', re.DOTALL)
        matches = pattern.findall(prompt)
        if matches:
            result = matches[-1].replace("问：", "").replace("\n答：", "").strip()
    elif re.search("neuralchat", model_name, re.IGNORECASE):
        matches = re.findall(r'### User:.*?### Assistant:', prompt, re.DOTALL)
        if matches:
            result = '''
### System:
    - You are a helpful assistant chatbot trained by Intel.
    - You answer questions.
    - You are excited to be able to help the user, \
but will refuse to do anything that could be considered harmful to the user.
    - You are more than just an information source, you are also able to write poetry,\
short stories, and make jokes.</s>
''' + matches[-1]

    return result

output_token_len = 0
def predict_stream(**params):
    """
    Generates streaming text based on the given parameters and prompt.

    Args:
        params (dict): A dictionary containing the parameters for text generation.
        `device` (string): Specifies the device type for text generation. It can be either "cpu", "cuda" or "hpu".
        `prompt` (string): Represents the initial input or context provided to the text generation model.
        `temperature` (float): Controls the randomness of the generated text.
                               Higher values result in more diverse outputs.
        `top_p` (float): Specifies the cumulative probability threshold for using in the top-p sampling strategy.
                         Smaller values make the output more focused.
        `top_k` (int): Specifies the number of highest probability tokens to consider in the top-k sampling strategy.
        `repetition_penalty` (float): Controls the penalty applied to repeated tokens in the generated text.
                                      Higher values discourage repetition.
        `max_new_tokens` (int): Limits the maximum number of tokens to be generated.
        `do_sample` (bool): Determines whether to use sampling-based text generation.
                            If set to True, the output will be sampled; otherwise,
                            it will be determined by the model's top-k or top-p strategy.
        `num_beams` (int): Controls the number of beams used in beam search.
                           Higher values increase the diversity but also the computation time.
        `model_name` (string): Specifies the name of the pre-trained model to use for text generation.
                               If not provided, the default model is "Intel/neural-chat-7b-v3-1".
        `num_return_sequences` (int): Specifies the number of alternative sequences to generate.
        `bad_words_ids` (list or None): Contains a list of token IDs that should not appear in the generated text.
        `force_words_ids` (list or None): Contains a list of token IDs that must be included in the generated text.
        `use_hpu_graphs` (bool): 
                    Determines whether to utilize Habana Processing Units (HPUs) for accelerated generation.
        `use_cache` (bool): Determines whether to utilize kv cache for accelerated generation.
        `ipex_int8` (bool): Whether to use IPEX int8 model to inference.

    Returns:
        generator: A generator that yields the generated streaming text.
    """
    start_time = datetime.now()
    device = params["device"] if "device" in params else "cpu"
    temperature = float(params["temperature"]) if "temperature" in params else 0.9
    top_p = float(params["top_p"]) if "top_p" in params else 0.75
    top_k = int(params["top_k"]) if "top_k" in params else 1
    repetition_penalty = (
        float(params["repetition_penalty"]) if "repetition_penalty" in params else 1.1
    )
    max_new_tokens = (
        int(params["max_new_tokens"]) if "max_new_tokens" in params else 256
    )
    do_sample = params["do_sample"] if "do_sample" in params else True
    num_beams = int(params["num_beams"]) if "num_beams" in params else 0
    model_name = (
        params["model_name"] if "model_name" in params else "Intel/neural-chat-7b-v3-1"
    )
    num_return_sequences = (
        params["num_return_sequences"] if "num_return_sequences" in params else 1
    )
    bad_words_ids = params["bad_words_ids"] if "bad_words_ids" in params else None
    force_words_ids = params["force_words_ids"] if "force_words_ids" in params else None
    use_hpu_graphs = params["use_hpu_graphs"] if "use_hpu_graphs" in params else False
    use_cache = params["use_cache"] if "use_cache" in params else True
    return_stats = params["return_stats"] if "return_stats" in params else False
    prompt = params["prompt"]
    ipex_int8 = params["ipex_int8"] if "ipex_int8" in params else False
    model = MODELS[model_name]["model"]
    tokenizer = MODELS[model_name]["tokenizer"]
    assistant_model = MODELS[model_name]["assistant_model"]
    errors_queue = Queue()
    if hasattr(model, 'device') and model.device.type != device:
        device = model.device.type

    if is_llm_runtime_model(model):
        prompt = remove_prompt_history(model_name, prompt)
        max_new_tokens = max_new_tokens if (max_new_tokens > 1024 or \
                                            "codellama" in model_name.lower() or \
                                            "starcoder" in model_name.lower() or \
                                            "codegen" in model_name.lower()) else 1024

    if is_llm_runtime_model(model):
        if "chatglm" in model_name.lower():
            prompt = tokenizer.build_prompt(prompt)
            input_tokens = tokenizer([prompt], return_tensors="pt").input_ids
            input_token_len = input_tokens.shape[-1]
            streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
        else:
            input_tokens, input_token_len = tokenization(prompt, tokenizer, device)
            streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    else:
        input_tokens, input_token_len = tokenization(prompt, tokenizer, device)
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
    if num_beams == 0:
        num_beams = 1
        do_sample = True

    generate_kwargs = get_generate_kwargs(
        max_new_tokens, input_token_len, 
        get_stop_token_ids(model, tokenizer),
        assistant_model=assistant_model
    )

    if device in ["cpu", "cuda", "xpu"]:
        if device in ["cuda", "xpu"]:
            input_tokens = prepare_inputs(
                input_tokens, model.device if hasattr(model, 'device') else torch.device(device)
            )
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            use_cache=use_cache,
            num_return_sequences=num_return_sequences,
        )

        def generate_output():
            dtype = model.dtype if hasattr(model, 'dtype') else torch.bfloat16
            try:
                with torch.no_grad():
                    if device == "cpu":
                        context = torch.cpu.amp.autocast(enabled=True, dtype=dtype, cache_enabled=True)
                    elif device == "cuda":
                        context = torch.cuda.amp.autocast(enabled=True, dtype=dtype, cache_enabled=True)
                    elif device == "xpu":
                        context = torch.xpu.amp.autocast(enabled=True, dtype=dtype, cache_enabled=True)
                    if ipex_int8:
                        global output_token_len
                        output_token=model.generate(
                                **input_tokens,
                                **generate_kwargs,
                                streamer=streamer,
                                generation_config=generation_config,
                                return_dict_in_generate=True,
                            )

                    else:
                        global output_token_len
                        if is_llm_runtime_model(model):  # optimized model gerenate
                            output_token=model.generate(
                                input_tokens if "chatglm" in model_name.lower() else input_tokens['input_ids'],
                                streamer=streamer,
                                temperature=temperature,
                                top_p=top_p,
                                top_k=top_k,
                                repetition_penalty=repetition_penalty,
                                max_new_tokens=max_new_tokens,
                                ctx_size=max_new_tokens,
                                ignore_prompt=True,
                                interactive=True,
                                do_sample=do_sample,
                                num_beams=num_beams,
                                n_keep=2 if "chatglm" in model_name.lower() else 1
                            )
                        else:
                            with context:
                                output_token=model.generate(
                                    **input_tokens,
                                    **generate_kwargs,
                                    streamer=streamer,
                                    generation_config=generation_config,
                                    return_dict_in_generate=True,
                                )
                    output_token_len= len(output_token[0]) if is_llm_runtime_model(model) else \
                                      output_token.sequences[0].shape[-1]
                    return output_token
            except Exception as e:
                errors_queue.put(e)

        generation_thread = Thread(target=generate_output)
        generation_thread.start()
    elif device == "hpu":
        # Move inputs to target device(s)
        input_tokens = prepare_inputs(input_tokens, model.device)

        # Generation configuration
        generation_config = copy.deepcopy(model.generation_config)
        generation_config.max_new_tokens = max_new_tokens
        generation_config.use_cache = use_cache
        generation_config.do_sample = do_sample
        generation_config.num_beams = num_beams
        generation_config.bad_words_ids = bad_words_ids
        generation_config.force_words_ids = force_words_ids
        generation_config.num_return_sequences = num_return_sequences
        generation_config.static_shapes = model_is_optimized(model.config)
        generation_config.top_k = top_k
        # TODO there is an issue when top_p is used in Habana
        # generation_config.top_p = top_p
        generation_config.temperature = temperature
        generation_config.repetition_penalty = repetition_penalty
        def generate_output():
            try:
                with torch.no_grad():
                    output_token=model.generate(
                        **input_tokens,
                        **generate_kwargs,
                        streamer=streamer,
                        generation_config=generation_config,
                        return_dict_in_generate=True,
                        output_scores=True,
                        max_new_tokens=max_new_tokens,
                        lazy_mode=True,
                        hpu_graphs=use_hpu_graphs,
                        ignore_eos=False,
                    )
                    global output_token_len
                    output_token_len=output_token.sequences[0].shape[-1]
                    return output_token
            except Exception as e:
                errors_queue.put(e)

        generation_thread = Thread(target=generate_output)
        generation_thread.start()
    else:
        raise ValueError(
            f"unsupported device type {device}, only supports cpu, xpu, cuda and hpu now."
        )
    output_word_len = 0

    generation_thread.join(0.1)
    if generation_thread.is_alive():
        pass
    else:
        thread_exception = errors_queue.get()
        raise thread_exception
    # prevent crash if no words are coming out
    first_word_output_time = datetime.now()
    for new_text in streamer:
        if len(new_text) == 0:
            continue
        if output_word_len == 0:
            first_word_output_time = datetime.now()
        output_word_len += 1
        yield new_text

    end_time = datetime.now()

    time.sleep(0.1)
    duration = int((end_time - start_time).total_seconds() * 1000)
    first_token_latency = int(
        (first_word_output_time - start_time).total_seconds() * 1000 * 3/4
    )
    if is_llm_runtime_model(model):
        msecond_per_token = (
            duration  / output_token_len
            if output_token_len != 1
            else
            0
        )
    else:
        msecond_per_token = (
            duration  / (output_token_len - input_token_len)
            if output_token_len != 1
            else
            0
        )
    if return_stats:
        stats = {
            "input_token_len": str(input_token_len),
            "output_token_len": str(output_token_len),
            "duration": str(duration) + " ms",
            "first_token_latency": str(first_token_latency) + " ms",
            "msecond_per_token": str(msecond_per_token) + " ms",
        }
        yield "\n| {:<22} | {:<27} |\n".format("Key", "Value")
        yield "| " + "-"*22 + " | " + "-"*27 + " |" + "\n"
        for key, value in stats.items():
            yield "| {:<22} | {:<27} |\n".format(key, value)


def predict(**params):
    """
    Generates streaming text based on the given parameters and prompt.

    Args:
        params (dict): A dictionary containing the parameters for text generation.
        `device` (string): Specifies the device type for text generation. It can be either "cpu", "cuda" or "hpu".
        `prompt` (string): Represents the initial input or context provided to the text generation model.
        `temperature` (float): Controls the randomness of the generated text.
                               Higher values result in more diverse outputs.
        `top_p` (float): Specifies the cumulative probability threshold for using in the top-p sampling strategy.
                         Smaller values make the output more focused.
        `top_k` (int): Specifies the number of highest probability tokens to consider in the top-k sampling strategy.
        `repetition_penalty` (float): Controls the penalty applied to repeated tokens in the generated text.
                                      Higher values discourage repetition.
        `max_new_tokens` (int): Limits the maximum number of tokens to be generated.
        `do_sample` (bool): Determines whether to use sampling-based text generation.
                            If set to True, the output will be sampled; otherwise,
                            it will be determined by the model's top-k or top-p strategy.
        `num_beams` (int): Controls the number of beams used in beam search.
                           Higher values increase the diversity but also the computation time.
        `model_name` (string): Specifies the name of the pre-trained model to use for text generation.
                               If not provided, the default model is "mosaicml/mpt-7b-chat".
        `num_return_sequences` (int): Specifies the number of alternative sequences to generate.
        `bad_words_ids` (list or None): Contains a list of token IDs that should not appear in the generated text.
        `force_words_ids` (list or None): Contains a list of token IDs that must be included in the generated text.
        `use_hpu_graphs` (bool): 
                 Determines whether to utilize Habana Processing Units (HPUs) for accelerated generation.
        `use_cache` (bool): Determines whether to utilize kv cache for accelerated generation.
        `ipex_int8` (bool): Whether to use IPEX int8 model to inference.

    Returns:
        generator: A generator that yields the generated streaming text.
    """
    device = params["device"] if "device" in params else "cpu"
    temperature = float(params["temperature"]) if "temperature" in params else 0.9
    top_p = float(params["top_p"]) if "top_p" in params else 0.75
    top_k = int(params["top_k"]) if "top_k" in params else 1
    repetition_penalty = (
        float(params["repetition_penalty"]) if "repetition_penalty" in params else 1.1
    )
    max_new_tokens = (
        int(params["max_new_tokens"]) if "max_new_tokens" in params else 256
    )
    do_sample = params["do_sample"] if "do_sample" in params else True
    num_beams = int(params["num_beams"]) if "num_beams" in params else 0
    model_name = (
        params["model_name"] if "model_name" in params else "mosaicml/mpt-7b-chat"
    )
    num_return_sequences = (
        params["num_return_sequences"] if "num_return_sequences" in params else 1
    )
    bad_words_ids = params["bad_words_ids"] if "bad_words_ids" in params else None
    force_words_ids = params["force_words_ids"] if "force_words_ids" in params else None
    use_hpu_graphs = params["use_hpu_graphs"] if "use_hpu_graphs" in params else False
    use_cache = params["use_cache"] if "use_cache" in params else False
    ipex_int8 = params["ipex_int8"] if "ipex_int8" in params else False
    prompt = params["prompt"]
    model = MODELS[model_name]["model"]
    tokenizer = MODELS[model_name]["tokenizer"]
    assistant_model=MODELS[model_name]["assistant_model"]
    if hasattr(model, "device") and model.device.type != device:
        device = model.device.type

    if is_llm_runtime_model(model):
        prompt = remove_prompt_history(model_name, prompt)
        max_new_tokens = max_new_tokens if (max_new_tokens > 1024 or \
                                            "codellama" in model_name.lower() or \
                                            "starcoder" in model_name.lower() or \
                                            "codegen" in model_name.lower()) else 1024

    if num_beams == 0:
        num_beams = 1
        do_sample = True

    input_tokens, input_token_len = tokenization(prompt, tokenizer, device)
    generate_kwargs = get_generate_kwargs(
        max_new_tokens, input_token_len, 
        get_stop_token_ids(model, tokenizer), 
        assistant_model=assistant_model
    )

    if device in ["cpu", "cuda", "xpu"]:
        if device in ["cuda", "xpu"]:
            input_tokens = prepare_inputs(
                input_tokens, model.device if hasattr(model, 'device') else torch.device(device)
            )
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            use_cache=use_cache,
            num_return_sequences=num_return_sequences,
        )
        dtype = model.dtype if hasattr(model, 'dtype') else torch.bfloat16
        with torch.no_grad():
            if device == "cpu":
                context = torch.cpu.amp.autocast(enabled=True, dtype=dtype, cache_enabled=True)
            elif device == "cuda":
                context = torch.cuda.amp.autocast(enabled=True, dtype=dtype, cache_enabled=True)
            elif device == "xpu":
                context = torch.xpu.amp.autocast(enabled=True, dtype=dtype, cache_enabled=True)
            if ipex_int8:
                generation_output = model.generate(
                        **input_tokens,
                        **generate_kwargs,
                        generation_config=generation_config,
                        return_dict_in_generate=True
                        )
            else:
                with context:
                    if is_llm_runtime_model(model):  # optimized model gerenate
                        generation_output = model.generate(
                            input_tokens['input_ids'],
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            repetition_penalty=repetition_penalty,
                            max_new_tokens=max_new_tokens,
                            ctx_size=max_new_tokens,
                            ignore_prompt=True,
                            interactive=True,
                            do_sample=do_sample,
                            num_beams=num_beams,
                            seed=1
                        )
                    else:
                        generation_output = model.generate(
                            **input_tokens,
                            **generate_kwargs,
                            generation_config=generation_config,
                            return_dict_in_generate=True
                        )
    elif device == "hpu":
        # Move inputs to target device(s)
        input_tokens = prepare_inputs(input_tokens, model.device)

        # Generation configuration
        generation_config = copy.deepcopy(model.generation_config)
        generation_config.max_new_tokens = max_new_tokens
        generation_config.use_cache = use_cache
        generation_config.do_sample = do_sample
        generation_config.num_beams = num_beams
        generation_config.bad_words_ids = bad_words_ids
        generation_config.force_words_ids = force_words_ids
        generation_config.num_return_sequences = num_return_sequences
        generation_config.static_shapes = model_is_optimized(model.config)
        generation_config.top_k = top_k
        # TODO there is an issue when top_p is used in Habana
        # generation_config.top_p = top_p
        generation_config.temperature = temperature
        generation_config.repetition_penalty = repetition_penalty

        with torch.no_grad():
            generation_output = model.generate(
                **input_tokens,
                **generate_kwargs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                lazy_mode=True,
                hpu_graphs=use_hpu_graphs,
                ignore_eos=False,
            )
    if is_llm_runtime_model(model):  # optimized model gerenate
        output = tokenizer.decode(generation_output[0], skip_special_tokens=True)
    else:
        output = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
    if "### Response:" in output:
        return output.split("### Response:")[1].strip()
    if "### Assistant" in output:
        return output.split("### Assistant:")[1].strip()
    if "\nassistant\n" in output:
        return output.split("\nassistant\n")[1].strip()
    if "[/INST]" in output:
        return output.split("[/INST]")[1].strip()
    if "答：" in output:
        return output.split("答：")[1].strip()
    return output
