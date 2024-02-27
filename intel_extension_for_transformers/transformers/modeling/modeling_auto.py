# !/usr/bin/env python
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

# coding=utf-8
# Copyright 2021 The EleutherAI and HuggingFace Teams. All rights reserved.
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

import json
import os
import re
import torch
import transformers
import types

from ..utils import (
    BitsAndBytesConfig,
    MixedPrecisionConfig,
    SmoothQuantConfig,
    WeightOnlyQuantConfig,
    logger,
    LazyImport,
)
from ..utils.utility import (
    generate_dummy_past_key_values,
    generate_dummy_past_key_values_for_opt_llm,
    MODEL_TYPES_REQUIRING_POSITION_IDS,
    IPEX_OPT_LLM_SUPPORTED,
    QUANT_CONFIG,
    WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
)
from ...llm.quantization.utils import (
    convert_dtype_str2torch,
    convert_dtype_torch2str,
    convert_to_quantized_model,
    replace_linear
)
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import is_accelerate_available, is_bitsandbytes_available
from typing import Union

torch = LazyImport("torch")


def convert_model_to_public(model):
    from intel_extension_for_pytorch.nn.utils._quantize_convert import WeightOnlyLinear  # pylint: disable=E0401
    for name, module in model.named_modules():
        if isinstance(module, WeightOnlyLinear):
            if module.weight_transposed:
                module.qweight.data = module.qweight.t_().contiguous()
                module.scales.data = module.scales.t_().contiguous()
                module.weight_transposed = False


def save_low_bit(
    self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs
):
    assert hasattr(
        self, "quantization_config"
    ), f"Detected this model is not a low-bit model."

    if os.path.isfile(save_directory):
        logger.error(
            f"Provided path ({save_directory}) should be a directory, not a file"
        )
        return

    # reorder weight and scales if they have been transposed
    if self.quantization_config.device == "xpu":
        convert_model_to_public(self)

    os.makedirs(save_directory, exist_ok=True)
    # use transformers original `save_pretrained` function
    del self.save_pretrained
    self.save_pretrained(
        save_directory=save_directory, push_to_hub=push_to_hub, **kwargs
    )
    self.save_pretrained = types.MethodType(save_low_bit, self)
    # We conveniently save all the keys of the model to have them on hand,
    # so that when using 'low_cpumem load',
    # it's not necessary to load the entire model to extract its keys
    # and we can avoid gc not triggered potentially.
    all_checkpoint_keys = {"all_checkpoint_keys": list(self.state_dict().keys())}
    json_file_path = os.path.join(save_directory, "all_checkpoint_keys.json")
    with open(json_file_path, "w") as json_file:
        json.dump(all_checkpoint_keys, json_file)
    if push_to_hub:
        use_auth_token = kwargs.pop("use_auth_token", None)

        if use_auth_token is not None:
            logger.warning.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        if token is not None:
            kwargs["token"] = token
        commit_message = kwargs.pop("commit_message", None)
        repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
        repo_id = self._create_repo(repo_id, **kwargs)
        files_timestamps = self._get_files_timestamps(save_directory)
        self._upload_modified_files(
            save_directory,
            repo_id,
            files_timestamps,
            commit_message=commit_message,
            token=kwargs.get("token"),
        )

    self.quantization_config.low_bit_model = True
    self.quantization_config.save_pretrained(save_directory, **kwargs)


class _BaseQBitsAutoModelClass:
    ORIG_MODEL = None
    model_type_list = ["llama", "gptj", "mpt", "opt", "gptneox",   \
            "dolly", "polyglot", "starcoder", "falcon", \
            "bloom", "chatglm2", "chatglm", "baichuan", \
            "mistral", "qwen", "phi", "whisper"]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model_file = kwargs.pop("model_file", None)
        if model_file is not None:
            from neural_speed import Model
            from huggingface_hub import hf_hub_download

            logger.info("Using Neural Speed to load the GGUF model...")

            gguf_model_file = hf_hub_download(pretrained_model_name_or_path, filename=model_file)

            if kwargs.get("model_type", False):
                model_type = kwargs.get("model_type")
            else:
                model_config = hf_hub_download(pretrained_model_name_or_path, filename="config.json")
                with open(model_config, "r", encoding="utf-8") as f:
                    hparams = json.load(f)
                    if "model_type" in hparams:
                        model_type = hparams["model_type"]
                    else:
                        logger.error("Can't get model_type from this Hugginface repo.")
                        exit(0)
            logger.info("The model_type is {}".format(model_type))

            if model_type not in cls.model_type_list:
                logger.error(
                    "Can't support this model_type. Please set the correct model_type, supported model_type: {}".format(
                        cls.model_type_list))
                exit(0)

            model = Model()
            model.init_from_bin(model_type, gguf_model_file)
            return model

        if kwargs.pop("use_embedding_runtime", False):
            from intel_extension_for_transformers.llm.runtime.deprecated.compile.graph import Graph
            from intel_extension_for_transformers.llm.runtime.deprecated.compile import compile, autocast

            cast_type = kwargs.get("cast_type", "native")
            with autocast(cast_type):
                model = compile(pretrained_model_name_or_path)

            return model

        device_map = kwargs.get("device_map", "cpu")
        use_cpu = (True if device_map == torch.device("cpu") or device_map == "cpu" else False)
        use_xpu = (True if device_map == torch.device("xpu") or device_map == "xpu" else False)

        if kwargs.get("use_llm_runtime", None) is not None:
            use_neural_speed = kwargs.pop("use_llm_runtime", True) and not use_xpu
            logger.warning("use_llm_runtime is deprecated in version 1.3.2, please use_neural_speed instead.")
        elif kwargs.get("use_neural_speed", None) is not None:
            use_neural_speed = kwargs.pop("use_neural_speed", True) and not use_xpu
        else:
            config = transformers.AutoConfig.from_pretrained(pretrained_model_name_or_path)
            if hasattr(config, "model_type") == False:
                logger.error("Can't get the model_type. Please check the correct model_type")
                exit(0)

            if config.model_type in cls.model_type_list:
                logger.info("Using Neural Speed...")
                use_neural_speed = True
            else:
                logger.info("Using Pytorch...")
                use_neural_speed = False

        if os.path.isfile(os.path.join(pretrained_model_name_or_path, QUANT_CONFIG)):
            logger.info(
                "Find quantization_config.json, trying to load quantized low bit model..."
            )
            quantization_config = WeightOnlyQuantConfig.from_pretrained(
                pretrained_model_name_or_path,
                _configuration_file=QUANT_CONFIG,
                **kwargs,
            )
            if quantization_config is None or quantization_config.low_bit_model != True:
                logger.warning("Quantization_config loading failed. If you want to load saved "
                               "low bit model, please check your quantization_config.json.")
            else:
                logger.info(
                    "quantization_config: {}".format(
                        quantization_config.to_json_string()
                    )
                )
                try:
                    kwargs["device_map"] = \
                        quantization_config.device if hasattr(quantization_config, "device") else "auto"
                    model = cls.load_low_bit(pretrained_model_name_or_path, *model_args, **kwargs)
                    logger.info("Saved low bit model loading successfully. Other input args "
                                "will be ignored.")
                    return model
                except Exception as e:
                    logger.error(e)
                    logger.error("Saved low bit model loading failed, please check your model.")
                    exit(0)


        import intel_extension_for_transformers.transformers.modeling.modeling_map

        load_in_8bit = kwargs.pop("load_in_8bit", False)
        load_in_4bit = kwargs.pop("load_in_4bit", False)
        quantization_config = kwargs.pop("quantization_config", None)

        if isinstance(quantization_config, BitsAndBytesConfig):
            model = cls.ORIG_MODEL.from_pretrained(
                pretrained_model_name_or_path,
                quantization_config=quantization_config,
                *model_args,
                **kwargs,
            )
            return model
        if load_in_8bit or load_in_4bit:
            if (is_accelerate_available() and is_bitsandbytes_available() and not use_cpu and not use_xpu):
                model = cls.ORIG_MODEL.from_pretrained(
                    pretrained_model_name_or_path,
                    quantization_config=quantization_config,
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    *model_args,
                    **kwargs,
                )
                logger.info("WeightOnlyQuant bitsandbytes done.")
                return model
            logger.info("{} device is used.".format(device_map))
            if load_in_8bit or load_in_4bit or quantization_config is not None:
                torch_dtype = kwargs.get("torch_dtype", torch.float16 if use_xpu else torch.float32)
                if use_xpu:
                    assert torch_dtype == torch.float16, \
                        "Intel GPU only support torch.float16 now, will support other dtype in future!"
                    kwargs["torch_dtype"] = torch_dtype
            if load_in_4bit:
                if quantization_config is None:
                    if use_neural_speed:
                        # use wnf4_sfp32_cfp32_g32_sym by default
                        quantization_config = WeightOnlyQuantConfig(compute_dtype="fp32", weight_dtype="nf4")
                    else:
                        quantization_config = WeightOnlyQuantConfig(compute_dtype=convert_dtype_torch2str(torch_dtype),
                                                                    weight_dtype="nf4" if use_cpu else "int4_fullrange")
                else:
                    assert ("4" in quantization_config.weight_dtype
                            and convert_dtype_str2torch(quantization_config.compute_dtype) == torch_dtype
                            ), "Quantization_config.weight_dtype should be 'nf4', 'int4_fullrange', 'int4_clip',"
                    f"'fp4_e2m1' or 'fp4_e2m1_bnb' and compute_dtype should be {torch_dtype}."
            elif load_in_8bit:
                if quantization_config is None:
                    if use_neural_speed:
                        quantization_config = WeightOnlyQuantConfig(compute_dtype="bf16", weight_dtype="int8")
                    else:
                        quantization_config = WeightOnlyQuantConfig(compute_dtype=convert_dtype_torch2str(torch_dtype),
                                                                    weight_dtype="int8")
                else:
                    assert (
                        quantization_config.weight_dtype == "int8" and quantization_config.compute_dtype == torch_dtype
                    ), f"Quantization_config.weight_dtype should be 'int8' and compute_dtype should be {torch_dtype}."
        if isinstance(quantization_config, MixedPrecisionConfig):
            if (quantization_config.dtype == "float16" or quantization_config.dtype == "fp16"):
                kwargs["torch_dtype"] = torch.float16
            else:
                kwargs["torch_dtype"] = torch.bfloat16
            kwargs["low_cpu_mem_usage"] = True
            try:
                model = cls.ORIG_MODEL.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
                model.config.update({"low_cpu_mem_usage": True})
            except NotImplementedError:
                logger.info(
                    "Failed to load models with `low_cpu_mem_usage` specified, "
                    "will fall to traditional load method with higher memory consumption."
                )
                kwargs["low_cpu_mem_usage"] = False
                model = cls.ORIG_MODEL.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
                model.config.update({"low_cpu_mem_usage": False})
            model = model.to("cpu")
            model.config.update({"device": "cpu"})
            model.eval()
            logger.info("Mixed Precision done.")
        elif isinstance(quantization_config, WeightOnlyQuantConfig):
            logger.info("Applying Weight Only Quantization.")
            if use_neural_speed:
                logger.info("Using LLM runtime.")
                quantization_config.post_init_runtime()
                from neural_speed import Model

                model = Model()
                model.init(
                    pretrained_model_name_or_path,
                    weight_dtype=quantization_config.weight_dtype,
                    alg=quantization_config.scheme,
                    group_size=quantization_config.group_size,
                    scale_dtype=quantization_config.scale_dtype,
                    compute_dtype=quantization_config.compute_dtype,
                    use_ggml=quantization_config.use_ggml,
                    use_quant=quantization_config.use_quant,
                    use_gptq=quantization_config.use_gptq or \
                            quantization_config.algorithm.upper() == "GPTQ" or \
                            quantization_config.use_autoround,
                    use_awq=quantization_config.algorithm.upper() == "AWQ",
                )
                model.quantization_config = quantization_config
                return model
            else:
                if use_xpu:
                    # TODO: if low_cpu_mem_uasge is True, gptj will have accuracy issue on CPU device.
                    kwargs["low_cpu_mem_usage"] = True
                    kwargs["device_map"] = "auto"
                    try:
                        model = cls.ORIG_MODEL.from_pretrained(
                            pretrained_model_name_or_path,
                            torchscript=True
                            if quantization_config.algorithm in ["TEQ", "AWQ"] and not use_xpu else False,
                            *model_args,
                            **kwargs,
                        )
                        model.config.update({"low_cpu_mem_usage": True})
                    except NotImplementedError:
                        logger.info("Failed to load models with `low_cpu_mem_usage` specified, "
                                    "will fall to traditional load method with higher memory consumption.")
                        kwargs["low_cpu_mem_usage"] = False
                        model = cls.ORIG_MODEL.from_pretrained(
                            pretrained_model_name_or_path,
                            torchscript=True
                            if quantization_config.algorithm in ["TEQ", "AWQ"] and not use_xpu else False,
                            *model_args,
                            **kwargs,
                        )
                        model.config.update({"low_cpu_mem_usage": False})
                else:
                    model = cls.ORIG_MODEL.from_pretrained(
                        pretrained_model_name_or_path,
                        torchscript=True if quantization_config.algorithm in ["TEQ", "AWQ"] and not use_xpu else False,
                        *model_args,
                        **kwargs,
                    )
                model.eval()
                quantization_config.update({"device": "cpu"})
                if use_xpu:
                    import intel_extension_for_pytorch
                    assert hasattr(torch, "xpu") and torch.xpu.is_available(), "There is no xpu device in this system!"
                    quantization_config.update({"device": "xpu"})
                if (not torch.cuda.is_available() or device_map == "cpu"
                        or device_map == torch.device("cpu")) and model.config.model_type == "chatglm":
                    model = model.float()
                if use_cpu:
                    quantization_config.post_init()
                elif use_xpu:
                    quantization_config.post_init_xpu()
                model = convert_to_quantized_model(model, quantization_config, device=device_map)

            # add quantization_config and save_low_bit to pretrained model dynamically
            model.device_map = device_map
            model.quantization_config = quantization_config

            model.save_pretrained = types.MethodType(save_low_bit, model)
            logger.info("WeightOnlyQuant done.")
        elif isinstance(quantization_config, SmoothQuantConfig):
            try:
                import intel_extension_for_pytorch as ipex
            except ImportError:
                logger.warning("Please install Intel Extension for PyTorch to accelerate the model inference.")
            assert (ipex.__version__ >= "2.1.0+cpu"), "Please use Intel Extension for PyTorch >=2.1.0+cpu."
            model = cls.ORIG_MODEL.from_pretrained(
                pretrained_model_name_or_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float,
                torchscript=True,
                use_cache=True,
                *model_args,
                **kwargs,
            )

            if (not torch.cuda.is_available() or device_map == "cpu"
                    or device_map == torch.device("cpu")) and model.config.model_type == "chatglm":
                model = model.float()
            model.eval()
            model_type = model.config.model_type.replace("_", "-")
            if "falcon" in model_type:
                logger.warning("Please use transformers 4.33.3 if you would like to apply smoothquant to Falcon.")
            if "llama" in model_type and transformers.__version__ >= "4.36.0":
                quantization_config.ipex_opt_llm = False
            logger.info("Applying SmoothQuant.")
            # ipex.optimize_transformers
            if quantization_config.ipex_opt_llm is None:
                if model_type in IPEX_OPT_LLM_SUPPORTED:
                    quantization_config.ipex_opt_llm = True
                    logger.info("quantization_config.ipex_opt_llm set to True and ipex.optimize_transformers is used.")
                    logger.warning("The suggested transformers version is 4.31.0.")
                else:
                    quantization_config.ipex_opt_llm = False
            if quantization_config.ipex_opt_llm:
                qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping(alpha=0.5)
                model = ipex.optimize_transformers(
                    model.eval(),
                    quantization_config=qconfig,
                    dtype=torch.float32,
                    inplace=True,
                    deployment_mode=False,
                )
                model.eval()

            # past_key_values
            num_beams = quantization_config.num_beams
            if quantization_config.ipex_opt_llm:
                past_key_values = generate_dummy_past_key_values_for_opt_llm(config=model.config,
                                                                             input_bs=1,
                                                                             num_beams=num_beams)
            else:
                past_key_values = generate_dummy_past_key_values(config=model.config, input_bs=1)

            # calibration function
            calib_func = quantization_config.calib_func
            tokenizer = quantization_config.tokenizer
            if calib_func is None:
                if quantization_config.tokenizer is None:
                    logger.error("Please provide the tokenizer or provide calib_func directly," +
                                 " the following is how to get tokenizer. \n" +
                                 " from transformer import AutoTokenizer \n" +
                                 " tokenizer = AutoTokenizer.from_pretrained(model_name_or_path) \n")
                    exit(0)

                from datasets import load_dataset
                from torch.utils.data import DataLoader

                calib_dataset = quantization_config.calib_dataset
                calib_shuffle = quantization_config.calib_shuffle
                calib_iters = quantization_config.calib_iters
                calib_padding = quantization_config.calib_padding
                calib_len = quantization_config.calib_len
                calib_pad_val = quantization_config.calib_pad_val
                from torch.nn.functional import pad

                calib_dataset = load_dataset(
                    calib_dataset,
                    split="test" if calib_dataset in ["mbpp", "openai_humaneval"] else "train",
                )
                if calib_shuffle:
                    calib_dataset = calib_dataset.shuffle(seed=42)

                def tokenize_function(examples):
                    if "prompt" in examples:
                        example = tokenizer(examples["prompt"])
                    elif "text" in examples:
                        example = tokenizer(examples["text"])
                    elif "code" in examples:
                        example = tokenizer(examples["code"])
                    else:
                        logger.error("Please check dataset prompt identifier," +
                                     " NeelNanda/pile-10k is default used calibration dataset.")
                        exit(0)
                    return example

                def collate_batch(batch):
                    position_ids_padded = []
                    input_ids_padded = []
                    last_ind = []
                    attention_mask_padded = []
                    for text in batch:
                        input_ids = text["input_ids"]
                        if not calib_padding:
                            input_ids = (input_ids[:int(calib_len)] if len(input_ids) > int(calib_len) else input_ids
                                         )  # no_padding
                        else:
                            pad_len = calib_len - input_ids.shape[0]
                            input_ids = pad(input_ids, (0, pad_len), value=calib_pad_val)

                        last_ind.append(input_ids.shape[0] - 1)
                        if model_type in ["bloom", "qwen"]:
                            attention_mask = torch.ones(len(input_ids) + 1)
                            attention_mask[0] = 0
                        else:
                            attention_mask = torch.ones(len(input_ids))
                        position_ids = torch.arange(len(input_ids))
                        input_ids_padded.append(input_ids)
                        attention_mask_padded.append(attention_mask)
                        position_ids_padded.append(position_ids)
                    if model_type in MODEL_TYPES_REQUIRING_POSITION_IDS:
                        return (
                            {
                                "input_ids": torch.vstack(input_ids_padded),
                                "attention_mask": torch.vstack(attention_mask_padded),
                                "position_ids": torch.vstack(position_ids_padded),
                                "past_key_values": past_key_values,
                            },
                            torch.tensor(last_ind),
                        )
                    else:
                        return (
                            {
                                "input_ids": torch.vstack(input_ids_padded),
                                "attention_mask": torch.vstack(attention_mask_padded),
                                "past_key_values": past_key_values,
                            },
                            torch.tensor(last_ind),
                        )

                def collate_batch_for_chatglm(batch):
                    last_ind = []
                    for text in batch:
                        input_ids = torch.vstack([text["input_ids"]])
                        if re.search("THUDM/chatglm-6b", model.config.auto_map["AutoConfig"]):
                            input_ids = (input_ids[:, :calib_len] if input_ids.shape[1] > calib_len else input_ids)
                            eos = torch.tensor([130001, 130004]).repeat(1, 1)
                            input_ids = torch.cat((input_ids, eos), 1)
                        else:
                            input_ids = (input_ids[:, :calib_len] if input_ids.shape[1] > calib_len else input_ids)
                        prepared_inputs = model.prepare_inputs_for_generation(input_ids)
                        attention_mask = torch.ones_like(input_ids)
                        last_ind.append(input_ids.shape[1] - 1)
                    return (
                        {
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "position_ids": prepared_inputs["position_ids"],
                            "past_key_values": past_key_values,
                        },
                        torch.tensor(last_ind),
                    )

                tokenized_dataset = calib_dataset.map(tokenize_function, batched=True)
                tokenized_dataset.set_format(type="torch", columns=["input_ids"])
                if model_type == "chatglm":
                    calib_dataloader = DataLoader(
                        tokenized_dataset,
                        batch_size=1,
                        shuffle=False,
                        collate_fn=collate_batch_for_chatglm,
                    )
                else:
                    calib_dataloader = DataLoader(
                        tokenized_dataset,
                        batch_size=1,
                        shuffle=False,
                        collate_fn=collate_batch,
                    )

                def calib_func(model):
                    with torch.no_grad():
                        for i, (inputs, last_ind) in enumerate(calib_dataloader):
                            if i >= calib_iters:
                                break
                            if model_type in MODEL_TYPES_REQUIRING_POSITION_IDS:
                                model(
                                    input_ids=inputs["input_ids"],
                                    past_key_values=inputs["past_key_values"],
                                    position_ids=inputs["position_ids"],
                                    attention_mask=inputs["attention_mask"],
                                )
                            else:
                                model(
                                    input_ids=inputs["input_ids"],
                                    past_key_values=inputs["past_key_values"],
                                    attention_mask=inputs["attention_mask"],
                                )

                logger.info("The default calibration function is used, " +
                            "the calibration dataset is NeelNanda/pile-10k, " +
                            "batchsize is 1 and calibration iteration is 100.")
                calib_func = calib_func

            # example_inputs
            example_inputs = quantization_config.example_inputs
            if example_inputs is None:
                for i, (inputs, last_ind) in enumerate(calib_dataloader):
                    if model_type in MODEL_TYPES_REQUIRING_POSITION_IDS:
                        example_inputs = {
                            "input_ids": inputs["input_ids"],
                            "attention_mask": inputs["attention_mask"],
                            "position_ids": inputs["position_ids"],
                            "past_key_values": inputs["past_key_values"],
                        }
                    else:
                        example_inputs = {
                            "input_ids": inputs["input_ids"],
                            "attention_mask": inputs["attention_mask"],
                            "past_key_values": inputs["past_key_values"],
                        }
                    break

            # call inc sq
            from neural_compressor import PostTrainingQuantConfig, quantization

            conf = PostTrainingQuantConfig(
                backend=quantization_config.backend,  # default is ipex
                excluded_precisions=quantization_config.excluded_precisions,
                op_type_dict=quantization_config.op_type_dict,
                op_name_dict=quantization_config.op_name_dict,
                recipes=quantization_config.recipes,
                example_inputs=example_inputs,
            )
            model = quantization.fit(
                model,
                conf,
                calib_func=calib_func,
                calib_dataloader=calib_dataloader
                if quantization_config.recipes["smooth_quant_args"]["alpha"] == "auto" else None,
            )
            logger.info("SmoothQuant done.")
        else:
            model = cls.ORIG_MODEL.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
            if (not torch.cuda.is_available() or device_map == "cpu"
                    or device_map == torch.device("cpu")) and model.config.model_type == "chatglm":
                model = model.float()

            model.eval()
        return model

    @classmethod
    def load_low_bit(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load a low bit optimized model (including INT4, INT5 and INT8) from a saved ckpt.
        :param pretrained_model_name_or_path: str value, Path to load the optimized model ckpt.
        # :param optimize_model: boolean value, Whether to further optimize the low_bit llm model.
        #                        Default to be True.
        :return: a model instance
        """
        from transformers.modeling_utils import (
            no_init_weights,
            get_checkpoint_shard_files,
            _add_variant,
        )
        from transformers.dynamic_module_utils import (
            resolve_trust_remote_code,
            get_class_from_dynamic_module,
        )
        from transformers.models.auto.configuration_auto import AutoConfig
        from transformers.utils import (
            ContextManagers,
            cached_file,
            download_url,
            extract_commit_hash,
            is_remote_url,
        )
        from transformers.generation.configuration_utils import GenerationConfig
        from transformers.models.auto.auto_factory import _get_model_class
        from accelerate.big_modeling import init_empty_weights
        import copy

        # Autofactory
        kwargs_orig = copy.deepcopy(kwargs)
        # modules_to_not_convert = kwargs.pop("modules_to_not_convert", None)
        trust_remote_code = kwargs.get("trust_remote_code", None)
        # Maybe needed when extract_local_archive_file
        subfolder = kwargs.get("subfolder", "")
        variant = kwargs.get("variant", None)
        offload_folder = kwargs.get("offload_folder", None)
        offload_state_dict = kwargs.get("offload_state_dict", False)
        torch_dtype = kwargs.pop("torch_dtype", "auto")
        cache_dir = kwargs.get("cache_dir", None)
        force_download = kwargs.get("force_download", False)
        proxies = kwargs.get("proxies", None)
        resume_download = kwargs.get("resume_download", False)
        local_files_only = kwargs.get("local_files_only", False)
        token = kwargs.get("token", None)
        from_pipeline = kwargs.get("_from_pipeline", None)
        from_auto_class = kwargs.get("_from_auto", False)
        revision = kwargs.get("revision", "main")
        commit_hash = kwargs.get("_commit_hash", None)
        _fast_init = kwargs.get("_fast_init", True)
        device_map = kwargs.pop("device_map", "auto")

        user_agent = {
            "file_type": "model",
            "framework": "pytorch",
            "from_auto_class": from_auto_class,
        }
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        # if torch_dtype=auto was passed here, ensure to pass it on
        if kwargs_orig.get("torch_dtype", None) == "auto":
            kwargs["torch_dtype"] = "auto"

        quantization_config = WeightOnlyQuantConfig.from_pretrained(
            pretrained_model_name_or_path,
            _configuration_file=QUANT_CONFIG,
            **kwargs,
        )

        assert (quantization_config is not None), "Detect this model is not a low-bit model."
        kwargs["trust_remote_code"] = trust_remote_code
        config, kwargs = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            return_unused_kwargs=True,
            **kwargs,
        )

        if commit_hash is None:
            if not isinstance(config, PretrainedConfig):
                # We make a call to the config file first (which may be absent)
                # to get the commit hash as soon as possible.
                resolved_config_file = cached_file(
                    pretrained_model_name_or_path,
                    "config.json",
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    _raise_exceptions_for_missing_entries=False,
                    _raise_exceptions_for_connection_errors=False,
                )
                commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
            else:
                commit_hash = getattr(config, "_commit_hash", None)

        config_dict, _ = PretrainedConfig.get_config_dict(pretrained_model_name_or_path)
        low_cpu_mem_usage = config_dict.pop("low_cpu_mem_usage", True)

        has_remote_code = (hasattr(config, "auto_map") and cls.ORIG_MODEL.__name__ in config.auto_map)

        has_local_code = type(config) in cls.ORIG_MODEL._model_mapping.keys()
        trust_remote_code = resolve_trust_remote_code(
            trust_remote_code,
            pretrained_model_name_or_path,
            has_local_code,
            has_remote_code,
        )
        if has_remote_code and trust_remote_code:
            class_ref = config.auto_map[cls.ORIG_MODEL.__name__]
            model_class = get_class_from_dynamic_module(class_ref, pretrained_model_name_or_path, **kwargs)
            if os.path.isdir(pretrained_model_name_or_path):
                model_class.register_for_auto_class(cls.ORIG_MODEL.__name__)
            else:
                cls.ORIG_MODEL.register(config.__class__, model_class, exist_ok=True)
        elif type(config) in cls.ORIG_MODEL._model_mapping.keys():
            model_class = _get_model_class(config, cls.ORIG_MODEL._model_mapping)

        # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
        # index of the files.
        is_sharded = False
        sharded_metadata = None
        if pretrained_model_name_or_path is not None:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            is_local = os.path.isdir(pretrained_model_name_or_path)
            if is_local:
                if os.path.isfile(
                        os.path.join(
                            pretrained_model_name_or_path,
                            subfolder,
                            _add_variant(WEIGHTS_NAME, variant),
                        )):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path,
                        subfolder,
                        _add_variant(WEIGHTS_NAME, variant),
                    )
                elif os.path.isfile(
                        os.path.join(
                            pretrained_model_name_or_path,
                            subfolder,
                            _add_variant(WEIGHTS_INDEX_NAME, variant),
                        )):
                    # Load from a sharded PyTorch checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path,
                        subfolder,
                        _add_variant(WEIGHTS_INDEX_NAME, variant),
                    )
                    is_sharded = True
                elif os.path.isfile(
                        os.path.join(
                            pretrained_model_name_or_path,
                            subfolder,
                            _add_variant(SAFE_WEIGHTS_NAME, variant),
                        )):
                    # Load from a safetensors checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path,
                        subfolder,
                        _add_variant(SAFE_WEIGHTS_NAME, variant),
                    )
                elif os.path.isfile(
                        os.path.join(
                            pretrained_model_name_or_path,
                            subfolder,
                            _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                        )):
                    # Load from a safetensors checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path,
                        subfolder,
                        _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                    )
                    is_sharded = True
            elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
                archive_file = pretrained_model_name_or_path
                is_local = True
            elif is_remote_url(pretrained_model_name_or_path):
                filename = pretrained_model_name_or_path
                resolved_archive_file = download_url(pretrained_model_name_or_path)
            else:
                raise EnvironmentError(
                    f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it"
                    " from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                    f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                    f" directory containing a file named {_add_variant(WEIGHTS_NAME, variant)}")

            if is_local:
                logger.info(f"loading weights file {archive_file}")
                resolved_archive_file = archive_file
            else:
                logger.info(
                    f"loading weights file {filename} from cache at {resolved_archive_file}"
                )
        else:
            resolved_archive_file = None

        # We'll need to download and cache each checkpoint shard if the checkpoint is sharded.
        if is_sharded:
            # rsolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
            resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
                pretrained_model_name_or_path,
                resolved_archive_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                token=token,
                user_agent=user_agent,
                revision=revision,
                subfolder=subfolder,
                _commit_hash=commit_hash,
            )

        # set dtype to instantiate the model under:
        # 1. If torch_dtype is not None, we use that dtype
        # 2. If torch_dtype is "auto", we auto-detect dtype from the loaded state_dict,
        #    by checking its first weights entry that is of a floating type
        #    - we assume all floating dtype weights are of the same dtype
        # we also may have config.torch_dtype available, but we won't rely on it till v5
        dtype_orig = None

        if torch_dtype is not None:
            if isinstance(torch_dtype, str):
                if torch_dtype == "auto":
                    if (hasattr(config, "torch_dtype") and config.torch_dtype is not None):
                        torch_dtype = config.torch_dtype
                    else:
                        if is_sharded and "dtype" in sharded_metadata:
                            torch_dtype = sharded_metadata["dtype"]
                        else:
                            torch_dtype = torch.float32
                else:
                    assert (False), f'`torch_dtype` can be either `torch.dtype` or `"auto"`, but received {torch_dtype}'
            dtype_orig = model_class._set_default_torch_dtype(torch_dtype)

        # Pretrained Model
        init_contexts = [no_init_weights(_enable=_fast_init)]
        init_contexts.append(init_empty_weights())

        if low_cpu_mem_usage:
            with ContextManagers(init_contexts):
                model = model_class(config, *model_args, **kwargs)
        else:
            model = model_class(config, *model_args, **kwargs)

        model = replace_linear(
            model,
            quantization_config=quantization_config,
            device=device_map,
            empty_weights=True,
        )

        if is_sharded:
            loaded_state_dict_keys = sharded_metadata["all_checkpoint_keys"]
        else:
            with open(os.path.join(pretrained_model_name_or_path, "all_checkpoint_keys.json"), "r") as json_file:
                loaded_data = json.load(json_file)
            loaded_state_dict_keys = loaded_data["all_checkpoint_keys"]

        # restore default dtype
        if dtype_orig is not None:
            torch.set_default_dtype(dtype_orig)

        (
            model,
            missing_keys,
            unexpected_keys,
            mismatched_keys,
            offload_index,
            error_msgs,
        ) = model_class._load_pretrained_model(
            model,
            None,
            loaded_state_dict_keys,  # XXX: rename?
            resolved_archive_file,
            pretrained_model_name_or_path,
            sharded_metadata=sharded_metadata,
            _fast_init=_fast_init,
            low_cpu_mem_usage=low_cpu_mem_usage,
            offload_folder=offload_folder,
            offload_state_dict=offload_state_dict,
            dtype=torch_dtype,
            keep_in_fp32_modules=[],
        )

        # make sure token embedding weights are still tied if needed
        model.tie_weights()

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        # If it is a model with generation capabilities, attempt to load the generation config
        if model.can_generate():
            try:
                model.generation_config = GenerationConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    subfolder=subfolder,
                    **kwargs,
                )
            except (OSError, TypeError):
                pass
        for param in model.parameters():
            param.requires_grad_(False)
        if device_map == "xpu":
            model = model.to("xpu")
        model.quantization_config = quantization_config
        model.save_pretrained = types.MethodType(save_low_bit, model)
        return model


class AutoModelForCausalLM(_BaseQBitsAutoModelClass):
    ORIG_MODEL = transformers.AutoModelForCausalLM


class AutoModel(_BaseQBitsAutoModelClass):
    ORIG_MODEL = transformers.AutoModel


class AutoModelForSeq2SeqLM(_BaseQBitsAutoModelClass):
    ORIG_MODEL = transformers.AutoModelForSeq2SeqLM
