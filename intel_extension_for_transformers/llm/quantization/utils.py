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


import logging
import os
import torch
from accelerate import init_empty_weights
from neural_compressor import quantization
from neural_compressor.config import PostTrainingQuantConfig
from neural_compressor.adaptor.torch_utils.model_wrapper import WeightOnlyLinear


logger = logging.getLogger(__name__)


DTYPE_BITS_MAPPING = {
    "nf4": 4,
    "fp4_e2m1_bnb": 4,
    "fp4_e2m1": 4,
    "int4_fullrange": 4,
    "int4_clip": 4,
    "fp8_e5m2": 8,
    "fp8_e4m3": 8,
    "int8": 8
}


def unpack_weight(qweight, scales, qzeros, q_config):
    group_size = q_config['group_size']
    bits = q_config['bits']
    wf = torch.tensor([[ 0,  4,  8, 12, 16, 20, 24, 28]], dtype=torch.int32)
    zeros = torch.bitwise_right_shift(torch.unsqueeze(qzeros, 2).expand(-1, -1, 32 // bits),
                                    wf.unsqueeze(0)).to(torch.int16 if bits == 8 else torch.int8)
    torch.bitwise_and(zeros, (2 ** bits) - 1, out=zeros)

    zeros = zeros + 1
    # zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])
    zeros = zeros.reshape(scales.shape)

    # scales = scales
    # scales = scales.reshape(-1, 1, scales.shape[-1])

    weight = torch.bitwise_right_shift(torch.unsqueeze(qweight, 1).expand(-1, 32 // bits, -1),
                                    wf.unsqueeze(-1)).to(torch.int16 if bits == 8 else torch.int8)
    torch.bitwise_and(weight,(2 ** bits) - 1, out=weight)
    # int_weight = weight.reshape(-1, group_size, weight.shape[2])

    return weight, scales, zeros

def replace_linear(
        model,
        modules_to_not_convert=None,
        current_key_name=None,
        quantization_config=None,
        device="cpu",
        empty_weights=False
    ):
    if modules_to_not_convert is None:
        modules_to_not_convert = ["lm_head"]
    if quantization_config.llm_int8_skip_modules:
        modules_to_not_convert = modules_to_not_convert.extend(quantization_config.llm_int8_skip_modules)
    model, is_replaced = _replace_linear(
         model, modules_to_not_convert, current_key_name, quantization_config, device=device,
        empty_weights=empty_weights
    )

    if not is_replaced:
        logger.warning(
            "You are loading your model in 8bit or 4bit but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )

    return model


def convert_dtype_2_str(dtype):
    if dtype == torch.float32:
        string = "fp32"
    elif dtype == torch.bfloat16:
        string = "bf16"
    elif dtype == torch.int8:
        string = "int8"
    else:
        string = "Unspport dtype"
    return string

def permute_func(weights, n_head: int, n_head_kv: int):
    if n_head_kv is not None and n_head != n_head_kv:
        n_head //= n_head_kv
    return (weights.reshape(n_head_kv, 2, weights.shape[0] // n_head_kv // 2, *weights.shape[1:])
                .swapaxes(1, 2)
                .reshape(weights.shape))

def _replace_linear(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    is_replaced=False,
    device="cpu",
    empty_weights=False
):
    """
    Private method that wraps the recursion for module replacement.

    Returns the converted model and a boolean that indicates if the conversion has been successfully or not.
    """
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if (isinstance(module, torch.nn.Linear) or isinstance(module, WeightOnlyLinear)) and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            if not any(key in ".".join(current_key_name) for key in modules_to_not_convert):
                with init_empty_weights():
                    in_features = module.in_features
                    out_features = module.out_features
                    if device == "cpu" or device == torch.device("cpu"):
                        from .nn.modules import QuantizedLinearQBits  # TODO: QuantizedLinearINT4, QuantizedLinearINT8
                        model._modules[name] = QuantizedLinearQBits(
                            in_features,
                            out_features,
                            module.bias is not None,
                            compute_dtype=quantization_config.compute_dtype,
                            compress_statistics=False,
                            weight_dtype=quantization_config.weight_dtype,
                            scale_dtype=quantization_config.scale_dtype,
                            blocksize=quantization_config.group_size,
                            scheme=quantization_config.scheme
                        )
                    else:
                        raise Exception("{} device Unsupport weight only quantization!".format(device))
                    # if quantization_config.quantization_method() == "s8":
                    #     model._modules[name] = QuantizedLinearINT8(
                    #         in_features,
                    #         out_features,
                    #         module.bias is not None,
                    #         compress_statistics=False,
                    #         blocksize=quantization_config.group_size,
                    #         scheme=quantization_config.scheme
                    #     )
                    #     is_replaced = True
                    # else:
                    #     model._modules[name] = QuantizedLinearINT4(
                    #         in_features,
                    #         out_features,
                    #         module.bias is not None,
                    #         compute_dtype=quantization_config.compute_dtype,
                    #         compress_statistics=False,
                    #         quant_dtype=quantization_config.quant_dtype,
                    #         blocksize=quantization_config.group_size,
                    #         scheme=quantization_config.scheme
                    #     )
                    #     is_replaced = True
                    is_replaced = True
                    # Store the module class in case we need to transpose the weight later
                    model._modules[name].source_cls = type(module)
                    # Force requires grad to False to avoid unexpected errors
                    model._modules[name].requires_grad_(False)
                if not empty_weights:
                    if quantization_config.algorithm == "GPTQ":
                        # to do, auto get n_head, n_head_kv
                        n_head = 12
                        n_head_kv = n_head
                        int_weight, gptq_scales, gptq_zeros = unpack_weight(module.qweight, module.scales, module.qzeros, quantization_config.gptq_quantize_config)
                        model._modules[name].set_gptq_weights_bias(
                            int_weight, gptq_scales, gptq_zeros, module.g_idx, quantization_config, n_head=n_head, n_head_kv=n_head_kv, permute_func=permute_func, bias=None if module.bias is None else module.bias.data)
                    else:
                        model._modules[name].set_weights_bias(
                            module.weight.data, None if module.bias is None else module.bias.data
                        )

        if len(list(module.children())) > 0:
            _, is_replaced = _replace_linear(
                module,
                modules_to_not_convert,
                current_key_name,
                quantization_config,
                is_replaced=is_replaced,
                device=device,
                empty_weights=empty_weights,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, is_replaced


def convert_to_quantized_model(model, config, device="cpu"):
    calib_dataloader = config.calib_dataloader
    calib_func = config.calib_func
    calib_iters = config.calib_iters
    model_device = next(model.parameters()).device
    if calib_dataloader is None and config.algorithm in ['TEQ', 'AWQ', 'GPTQ']:
        from datasets import load_dataset
        from torch.utils.data import DataLoader

        calib_dataset = config.calib_dataset
        if isinstance(calib_dataset, (str, bytes, os.PathLike)):
            calib_dataset = load_dataset(calib_dataset, split="train")
        calib_dataset = calib_dataset.shuffle(seed=42)
        if config.tokenizer is None:
            logger.error(
                "Please provide the tokenizer or provide calib_func directly,"
                + " the following is how to get tokenizer. \n"
                + " from transformer import AutoTokenizer \n"
                + " tokenizer = AutoTokenizer.from_pretrained(model_name_or_path) \n"
            )
            exit(0)

        def tokenize_function(examples):
            if "prompt" in examples:
                example = config.tokenizer(examples["prompt"])
            elif "text" in examples:
                example = config.tokenizer(examples["text"])
            elif "code" in examples:
                example = config.tokenizer(examples["code"])
            else:
                logger.error(
                    "Please check dataset prompt identifier,"
                    + " NeelNanda/pile-10k is default used calibration dataset."
                )
                exit(0)
            return example

        tokenized_dataset = calib_dataset.map(tokenize_function, batched=True)
        tokenized_dataset.set_format(type="torch", columns=["input_ids"])

        def collate_batch(batch):
            input_ids_padded = []
            for text in batch:
                input_ids = text["input_ids"]
                input_ids = (
                    input_ids[:512] if len(input_ids) > 512 else input_ids
                )
                input_ids_padded.append(input_ids)
            return torch.vstack(input_ids_padded)
        calib_dataloader = DataLoader(
            tokenized_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_batch,
        )
    if calib_func is None and config.algorithm in ['AWQ']:
        def default_calib_func(model):
            """
            This is the default calibration function, the dataset is NeelNanda/pile-10k,
            the default calib_iters is 100.
            """
            for i, (input_ids) in enumerate(calib_dataloader):
                if i >= calib_iters:
                    break
                model(
                    input_ids=input_ids,
                )
        calib_func = default_calib_func
        logger.info(
            "The default calibration funcation is used, "
            + "the calibration dataset is NeelNanda/pile-10k,"
            + "batchsize is 1 and calibration iteration is 100."
        )
    if config.weight_dtype in ["fp8_e4m3", "fp8_e5m2"]:
        return replace_linear(model, None, None, config, device=device)
    else:
        bits = DTYPE_BITS_MAPPING[config.weight_dtype]
        if config.weight_dtype == "int8":
            dtype = "int8"
        elif "int4" in config.weight_dtype:
            dtype = "int4"
        else:
            dtype = config.weight_dtype
        recipes={
            "rtn_args":{"enable_full_range": True if "fullrange" in config.weight_dtype else False,
                        "enable_mse_search": config.mse_range},
        }
        if config.gptq_recipes is not None:
            recipes["gptq_args"] = config.gptq_recipes
        conf = PostTrainingQuantConfig(
            approach="weight_only",
            op_type_dict={
                ".*":{
                    "weight": {
                        "bits": bits,
                        "dtype":dtype,
                        "group_size": config.group_size,  # -1 (per-channel)
                        "scheme": config.scheme,
                        "algorithm": config.algorithm, 
                    },
                },
            },
            recipes=recipes
        )
        # TEQ: set calib_func=None, use default training func as calib_func
        # RTN: doesn't need calib_func
        if config.algorithm in ['TEQ','RTN','GPTQ']:
            calib_func=None
        inc_model = quantization.fit(model,
                                    conf,
                                    calib_func=calib_func,
                                    calib_dataloader=calib_dataloader)
        if config.algorithm == "GPTQ":
            inc_model = inc_model.export_compressed_model(use_optimum_format=True)
            quantize_config = {
                "bits": bits,
                "group_size": config.group_size,
                "damp_percent": config.gptq_recipes["percdamp"],
                "desc_act":  config.gptq_recipes["act_order"],
                "sym": True if config.scheme=="sym" else False,
                "true_sequential": True,
                "model_name_or_path": "null",
                "model_file_base_name": "model"
            }

            setattr(config, "gptq_quantize_config", quantize_config)
            return replace_linear(inc_model, None, None, config, device=device)

        return replace_linear(inc_model.model, None, None, config, device=device)

