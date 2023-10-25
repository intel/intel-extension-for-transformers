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
from datasets import load_dataset
from neural_compressor import quantization
from neural_compressor.config import PostTrainingQuantConfig
from transformers import default_data_collator
from torch.utils.data import DataLoader, RandomSampler


logger = logging.getLogger(__name__)


def replace_linear(
        model,
        modules_to_not_convert=None,
        current_key_name=None,
        quantization_config=None,
        device=torch.device("cpu")
):
    model.to(device)
    if modules_to_not_convert is None:
        modules_to_not_convert = []
    if quantization_config.llm_int8_skip_modules:
        modules_to_not_convert = modules_to_not_convert.extend(quantization_config.llm_int8_skip_modules)
    model, is_replaced = _replace_linear(
        model, modules_to_not_convert, current_key_name, quantization_config, device=device,
    )

    if not is_replaced:
        logger.warning(
            "You are loading your model in 8bit or 4bit but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )

    return model


def get_weight_type_from_config(config):
    if config.weight_dtype == "int8":
        if config.scale_dtype == "fp32":
            weight_type = "s8_scalef32"
        else:
            raise Exception("scale_dtype only support fp32 now!")
    elif config.weight_dtype == "int4_fullrange":
        if config.scale_dtype == "fp32":
            weight_type = "s4fullrange_scalef32"
        else:
            raise Exception("scale_dtype only support fp32 now!")
    elif config.weight_dtype == "int4_clip":
        if config.scale_dtype == "fp32":
            weight_type = "s4clip_scalef32"
        else:
            raise Exception("scale_dtype only support fp32 now!")
    elif config.weight_dtype == "fp4_e2m1_bnb":
        if config.scale_dtype == "fp32":
            weight_type = "fp4bnb_scalef32"
        else:
            raise Exception("scale_dtype only support fp32 now!")
    elif config.weight_dtype == "fp4_e2m1":
        if config.scale_dtype == "fp32":
            weight_type = "fp4e2m1_scalef32"
        else:
            raise Exception("scale_dtype only support fp32 now!")
    elif config.weight_dtype == "nf4":
        if config.scale_dtype == "fp32":
            weight_type = "nf4_scalef32"
        else:
            raise Exception("scale_dtype only support fp32 now!")
    return weight_type


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


def _replace_linear(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    is_replaced=False,
    device=torch.device("cpu")
):
    """
    Private method that wraps the recursion for module replacement.

    Returns the converted model and a boolean that indicates if the conversion has been successfull or not.
    """
    weight_dtype = get_weight_type_from_config(quantization_config)
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if isinstance(module, torch.nn.Linear) and name not in modules_to_not_convert:
            from .nn import QuantizedLinearCPU  # TODO: QuantizedLinearINT4, QuantizedLinearINT8
            # Check if the current key is not in the `modules_to_not_convert`
            if not any(key in ".".join(current_key_name) for key in modules_to_not_convert):
                with init_empty_weights():
                    in_features = module.in_features
                    out_features = module.out_features

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
                    if device == "cpu" or device == torch.device("cpu"):
                        model._modules[name] = QuantizedLinearCPU(
                            in_features,
                            out_features,
                            module.bias is not None,
                            compute_dtype=quantization_config.compute_dtype,
                            compress_statistics=False,
                            weight_dtype=weight_dtype,
                            blocksize=quantization_config.group_size,
                            scheme=quantization_config.scheme
                        )
                    elif device == "xpu" or device == torch.device("xpu"):
                        pass
                    else:
                        raise Exception("{} device Unsupport weight only quantization!".format(device))

                    is_replaced = True
                    # Store the module class in case we need to transpose the weight later
                    model._modules[name].source_cls = type(module)
                    # Force requires grad to False to avoid unexpected errors
                    model._modules[name].requires_grad_(False)
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
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, is_replaced

def convert_to_quantized_model(model, config, device=torch.device("cpu")):
    if device == "xpu" or device == torch.device("xpu"):
        import intel_extension_for_pytorch
        assert hasattr(torch, "xpu") and torch.xpu.is_available(), "There is no xpu device in this system!"
    calib_dataloader = config.calib_dataloader
    calib_func = config.calib_func
    calib_iters = config.calib_iters
    model_device = next(model.parameters()).device
    if calib_dataloader is None and config.algorithm in ['TEQ', 'AWQ']:
        calib_dataset = config.calib_dataset
        if isinstance(calib_dataset, (str, bytes, os.PathLike)):
            calib_dataset = load_dataset(calib_dataset, split="train")
        calib_dataset = calib_dataset.shuffle(seed=42)
        generator = torch.Generator()
        generator.manual_seed(10)
        sampler = RandomSampler(calib_dataset, generator=generator)
        calib_dataloader = DataLoader(
            calib_dataset,
            batch_size=1,
            sampler=sampler,
            shuffle=False,
            collate_fn=default_data_collator,
        )
    if calib_func is None and config.algorithm in ['AWQ']:
        def default_calib_func(model):
            """
            This is the default calibration function, the dataset is NeelNanda/pile-10k,
            the default calib_iters is 100.
            """
            for i, (input) in enumerate(calib_dataloader):
                if i >= calib_iters:
                    break
                if isinstance(input, (tuple, list)):
                    for key in input[0]:
                        input[0][key] = input[0][key].to(model_device)
                    model(**input[0])
                elif isinstance(input, dict):
                    for key in input:
                        input[key] = input[key].to(model_device)
                    model(**input)
        calib_func = default_calib_func
        logger.info(
            "The default calibration funcation is used, "
            + "the calibration dataset is NeelNanda/pile-10k,"
            + "batchsize is 1 and calibration iteration is 100."
        )
    bits = 1  # only for int8
    if config.weight_dtype == "int8":
        dtype = "int8"
        bits = 8
    elif "int4" in config.weight_dtype:
        dtype = "int4"
    else:
        dtype = config.weight_dtype
    conf = PostTrainingQuantConfig(
        approach="weight_only",
        op_type_dict={
            ".*":{
                "weight": {
                    "bits": bits,
                    "dtype": dtype,
                    "group_size": config.group_size,  # -1 (per-channel)
                    "scheme": config.scheme,
                    "algorithm": config.algorithm,
                },
            },
        },
        recipes={
            "rtn_args": {"enable_full_range": True if "fullrange" in config.weight_dtype else False,
                         "enable_mse_search": config.mse_range},
        },
    )
    # TEQ: set calib_func=None, use default training func as calib_func
    # RTN: doesn't need calib_func
    if config.algorithm in ['TEQ', 'RTN']:
        calib_func = None
    inc_model = quantization.fit(model,
                                 conf,
                                 calib_func=calib_func,
                                 calib_dataloader=calib_dataloader)
    return replace_linear(inc_model.model, None, None, config, device=device)

def convert_dtype_str2torch(str_dtype):
    if str_dtype == "int8":
        return torch.int8
    elif str_dtype == "fp32":
        return torch.float
    elif str_dtype == "fp16":
        return torch.float16
    elif str_dtype == "bf16":
        return torch.bfloat16
    else:
        assert False, "Unsupport dtype {} for by IPEX backend".format(str_dtype)

def convert_to_quantized_model_by_ipex(model, config, device=torch.device("cpu")):
    import intel_extension_for_pytorch as ipex
    tmp_quan_weight_path = "./itrex_tmp_quantized_weight.pt"

    if config.weight_dtype == "int8":
        bits = 8
    elif "int4" in config.weight_dtype:
        bits = 4
    else:
        assert False, "Unsupport {} for quantize weight only by IPEX backend".format(config.weight_dtype)

    amp_dtype = convert_dtype_str2torch(config.compute_dtype)
    dataloader = config.calib_dataloader
    assert dataloader is not None, "Must provide config.calib_dataloader"

    ipex.woq(model,
        dataloader,
        tmp_quan_weight_path,
        wbits=bits,
        mixed_weight=True,
        group_size = config.group_size,
        param_dtype=amp_dtype)

    woq_config = {}
    woq_config['is_int4'] = (bits == 4)
    woq_config['group_size'] = config.group_size
    woq_config['weight_path'] = tmp_quan_weight_path
    amp_dtype = torch.float16
    model = ipex.optimize_transformers(model.eval(), dtype=amp_dtype, **woq_config)

    model.to(device)

    return model