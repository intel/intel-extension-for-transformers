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
import gc
import math
import os
from accelerate import init_empty_weights
from datasets import load_dataset
from neural_compressor import quantization
from neural_compressor.adaptor.torch_utils.model_wrapper import WeightOnlyLinear
from neural_compressor.utils.utility import LazyImport
from neural_compressor.config import PostTrainingQuantConfig
from ...utils.utils import is_ipex_available
from transformers import AutoTokenizer

if is_ipex_available():
    import intel_extension_for_pytorch as ipex


torch = LazyImport("torch")


logger = logging.getLogger(__name__)


DTYPE_BITS_MAPPING = {
    "nf4": 4,
    "fp4_e2m1_bnb": 4,
    "fp4_e2m1": 4,
    "int4_fullrange": 4,
    "int4_clip": 4,
    "fp8_e5m2": 8,
    "fp8_e4m3": 8,
    "int8": 8,
}


def replace_linear(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    device="cpu",
    empty_weights=False,
):
    if modules_to_not_convert is None:
        modules_to_not_convert = ["lm_head"]
    if quantization_config.llm_int8_skip_modules:
        modules_to_not_convert = modules_to_not_convert.extend(
            quantization_config.llm_int8_skip_modules
        )
    model, is_replaced = _replace_linear(
        model,
        modules_to_not_convert,
        current_key_name,
        quantization_config,
        device=device,
        empty_weights=empty_weights,
    )

    if not is_replaced:
        logger.warning(
            "You are loading your model in 8bit or 4bit but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )

    return model


def _replace_linear(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    is_replaced=False,
    device="cpu",
    empty_weights=False,
):
    """
    Private method that wraps the recursion for module replacement.

    Returns the converted model and a boolean that indicates if the conversion has been successfully or not.
    """
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)
        is_removed = False

        if (isinstance(module, torch.nn.Linear) or isinstance(module, WeightOnlyLinear)
            or (is_ipex_available() and isinstance(module, ipex.nn.utils._weight_prepack._IPEXLinear))) \
           and (name not in modules_to_not_convert):
            # Check if the current key is not in the `modules_to_not_convert`
            if not any(
                key in ".".join(current_key_name) for key in modules_to_not_convert
            ):
                with init_empty_weights():
                    in_features = module.in_features
                    out_features = module.out_features
                    if device == "cpu" or device == torch.device("cpu") or device == "auto":
                        from .nn.modules import (
                            QuantizedLinearQBits,
                        )  # TODO: QuantizedLinearINT4, QuantizedLinearINT8

                        model._modules[name] = QuantizedLinearQBits(
                            in_features,
                            out_features,
                            module.bias is not None,
                            compute_dtype=quantization_config.compute_dtype,
                            compress_statistics=False,
                            weight_dtype=quantization_config.weight_dtype,
                            scale_dtype=quantization_config.scale_dtype,
                            blocksize=quantization_config.group_size,
                            scheme=quantization_config.scheme,
                        )
                    elif device == "xpu" or device == torch.device("xpu"):
                        from intel_extension_for_pytorch.nn.utils._quantize_convert \
                            import WeightOnlyLinear as ipex_linear  # pylint: disable=E0401
                        model._modules[name] = ipex_linear(
                            in_features,
                            out_features,
                            module.bias is not None,
                            compute_dtype=quantization_config.compute_dtype,
                            compress_statistics=False,
                            weight_dtype=quantization_config.weight_dtype,
                            scale_dtype=quantization_config.scale_dtype,
                            blocksize=quantization_config.group_size,
                            scheme=quantization_config.scheme,
                            compression_dtype=module.compression_dtype
                            if hasattr(module, "compression_dtype") else torch.int8,
                            compression_dim=module.compression_dim if hasattr(module, "compression_dim") else 0,
                            device=device,
                            use_optimum_format=module.use_optimum_format
                            if hasattr(module, "use_optimum_format") else False,
                        )
                        if quantization_config.algorithm == "GPTQ":
                            g_idx = module.g_idx if hasattr(module, "g_idx") else \
                                torch.zeros(in_features, dtype=torch.int32).to(device)
                        else:
                            g_idx = None
                        model._modules[name].set_scales_zps_gidx(
                            module.scales if hasattr(module, "scales") else torch.ones(
                                    (out_features, math.ceil(in_features / quantization_config.group_size)),
                                    dtype=convert_dtype_str2torch(quantization_config.compute_dtype),
                                    device=torch.device(device)),
                            module.qzeros if hasattr(module, "qzeros") else None,
                            g_idx
                        )
                    else:
                        raise Exception("{} device Unsupported weight only quantization!".format(device))

                    is_replaced = True
                    # Store the module class in case we need to transpose the weight later
                    model._modules[name].source_cls = type(module)
                    # Force requires grad to False to avoid unexpected errors
                    model._modules[name].requires_grad_(False)
                if device == "cpu" or device == torch.device("cpu") or device == "auto":
                    if not empty_weights:
                        if quantization_config.algorithm == "GPTQ":
                            from .gptq_utils import unpack_weight
                            int_weight, gptq_scales, gptq_zeros = unpack_weight(
                                module.qweight,
                                module.scales,
                                module.qzeros,
                                quantization_config.gptq_quantize_config,
                            )
                            int_weight = int_weight.view(-1, int_weight.shape[-1])
                            model._modules[name].set_gptq_weights_bias(
                                int_weight,
                                gptq_scales,
                                gptq_zeros,
                                module.g_idx,
                                quantization_config,
                                bias=None if module.bias is None else module.bias.data,
                            )
                        else:
                            model._modules[name].set_weights_bias(
                                module.weight.data,
                                None if module.bias is None else module.bias.data,
                            )
                    else:
                        model._modules[name].set_weights_bias(
                            module.weight.data,
                            None if module.bias is None else module.bias.data,
                        )
                else:
                    if not hasattr(module, "qweight"):
                        n_pack = 8 // DTYPE_BITS_MAPPING[quantization_config.weight_dtype]
                        weight = torch.zeros(
                            (math.ceil(out_features / n_pack), in_features),
                            dtype=torch.int8, device=torch.device(device)
                        )
                    model._modules[name].set_weights_bias(
                        module.qweight.data if hasattr(module, "qweight") else weight,
                        None if module.bias is None else module.bias.data)
                    del module
                    gc.collect()
                    is_removed = True

        if not is_removed and len(list(module.children())) > 0: # pylint: disable=E1101
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
    if device == "xpu" or device == torch.device("xpu"):
        import intel_extension_for_pytorch
        assert hasattr(torch, "xpu") and torch.xpu.is_available(), "There is no xpu device in this system!"
    calib_dataloader = config.calib_dataloader
    calib_func = config.calib_func
    calib_iters = config.calib_iters
    model_device = next(model.parameters()).device
    if calib_dataloader is None and config.algorithm in ["TEQ", "AWQ", "GPTQ"]:
        from datasets import load_dataset
        from torch.utils.data import DataLoader

        calib_dataset = config.calib_dataset
        if isinstance(calib_dataset, (str, bytes, os.PathLike)):
            calib_dataset = load_dataset(calib_dataset, split="train")
        calib_dataset = calib_dataset.shuffle(seed=42)
        if config.tokenizer is None:
            logger.error(
                "Please provide the tokenizer or provide calib_func directly,"
                + " the following is how to get tokenizer. \n" +
                " from transformer import AutoTokenizer \n" +
                " tokenizer = AutoTokenizer.from_pretrained(model_name_or_path) \n"
            )
            exit(0)

        def tokenize_function(examples):
            if "prompt" in examples:
                example = config.tokenizer(examples["prompt"])
            elif "code" in examples:
                example = config.tokenizer(examples["code"])
            elif "text" in examples:
                example = config.tokenizer(examples["text"])
            else:
                logger.error(
                    "Please check dataset prompt identifier," +
                    " NeelNanda/pile-10k is default used calibration dataset.")
                exit(0)
            return example

        tokenized_dataset = calib_dataset.map(tokenize_function, batched=True)
        tokenized_dataset.set_format(type="torch", columns=["input_ids"])

        def collate_batch(batch):
            input_ids_padded = []
            for text in batch:
                input_ids = text["input_ids"]
                input_ids = input_ids[:512] if (len(input_ids) > 512 and config.algorithm != "GPTQ") else input_ids
                input_ids_padded.append(input_ids)
            return torch.vstack(input_ids_padded)

        calib_dataloader = DataLoader(
            tokenized_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_batch,
        )
    if calib_func is None and config.algorithm in ["AWQ"]:

        def default_calib_func(model):
            """
            This is the default calibration function, the dataset is NeelNanda/pile-10k,
            the default calib_iters is 100.
            """
            for i, (input_ids) in enumerate(calib_dataloader):
                if i >= calib_iters:
                    break
                model(input_ids=input_ids, )

        calib_func = default_calib_func
        logger.info("The default calibration function is used, " +
                    "the calibration dataset is NeelNanda/pile-10k," +
                    "batchsize is 1 and calibration iteration is 100.")
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
        recipes = {
            "rtn_args": {
                "enable_full_range": True
                if "fullrange" in config.weight_dtype
                else False,
                "enable_mse_search": config.mse_range,
            },
            "awq_args": config.algorithm_args.update({"enable_mse_search": config.mse_range})
                if config.algorithm == "AWQ" and config.algorithm_args is not None else {},
            "gptq_args": config.algorithm_args if config.algorithm == "GPTQ" else None
        }
        conf = PostTrainingQuantConfig(
            approach="weight_only",
            op_type_dict={
                ".*": {
                    "weight": {
                        "bits": bits,
                        "dtype": dtype,
                        "group_size": config.group_size,  # -1 (per-channel)
                        "scheme": config.scheme,
                        "algorithm": config.algorithm,
                    },
                },
            },
            op_name_dict={
                '.*lm_head': {  # re.match
                    "weight": {
                        'dtype': 'fp32'
                    },
                },
            },
            recipes=recipes,
        )
        # TEQ: set calib_func=None, use default training func as calib_func
        # RTN: doesn't need calib_func
        if config.algorithm in ["TEQ", "RTN", "GPTQ"]:
            calib_func = None

        orig_dtype = torch.float32
        for param in model.parameters():
            orig_dtype = param.dtype
            if orig_dtype != torch.float32:
                model.to(dtype=torch.float32)
            break

        inc_model = quantization.fit(model,
                                     conf,
                                     calib_func=calib_func,
                                     calib_dataloader=calib_dataloader)
        if device == "xpu" or device == torch.device("xpu"):
            model = inc_model.export_compressed_model(compression_dtype=torch.int8,
                                                      compression_dim=0,
                                                      use_optimum_format=False,
                                                      scale_dtype=convert_dtype_str2torch(config.scale_dtype))
            q_model = replace_linear(model,
                                     None,
                                     None,
                                     config,
                                     device=device)
        else:
            if config.algorithm == "GPTQ":
                inc_model = inc_model.export_compressed_model(use_optimum_format=True)
                inc_model.eval()

                quantize_config = {
                    "bits": bits,
                    "group_size": config.group_size,
                    "damp_percent": config.algorithm_args["percdamp"],
                    "desc_act": config.algorithm_args["act_order"],
                    "sym": True if config.scheme == "sym" else False,
                    "true_sequential": True,
                    "model_name_or_path": "null",
                    "model_file_base_name": "model",
                }

                setattr(config, "gptq_quantize_config", quantize_config)
                q_model = replace_linear(inc_model, None, None, config, device=device)
            else:
                q_model = replace_linear(inc_model.model, None, None, config, device=device)
        if orig_dtype != torch.float32:
            q_model.to(dtype=orig_dtype)
        return q_model.to(device)

def convert_dtype_str2torch(str_dtype):
    if str_dtype == "int8":
        return torch.int8
    elif str_dtype == "fp32" or str_dtype == "auto":
        return torch.float
    elif str_dtype == "fp16":
        return torch.float16
    elif str_dtype == "bf16":
        return torch.bfloat16
    else:
        assert False, "Unsupported str dtype {} to torch dtype".format(str_dtype)


def convert_dtype_torch2str(dtype):
    if dtype == torch.int8:
        return "int8"
    elif dtype == torch.float:
        return "fp32"
    elif dtype == torch.float16:
        return "fp16"
    elif dtype == torch.bfloat16:
        return "bf16"
    elif isinstance(dtype, str) and dtype in ["int8", "fp32", "fp16", "bf16"]:
        return dtype
    else:
        assert False, "Unsupported pytorch dtype {} to str dtype".format(dtype)


def get_bits(config):
    if config.weight_dtype == "int8":
        bits = 8
    elif "int4" in config.weight_dtype:
        bits = 4
    else:
        assert False, "Unsupported {} for quantize weight only by IPEX backend".format(config.weight_dtype)
    return bits
