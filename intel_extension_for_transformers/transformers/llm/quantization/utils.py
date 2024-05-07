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
from ...utils import CpuInfo
from accelerate import init_empty_weights
from datasets import load_dataset
from neural_compressor import quantization
from neural_compressor.adaptor.torch_utils.model_wrapper import WeightOnlyLinear
from neural_compressor.utils.utility import LazyImport
from neural_compressor.torch.algorithms.weight_only.autoround import get_autoround_default_run_fn
from neural_compressor.torch.quantization import (
    AutoRoundConfig,
    RTNConfig,
    GPTQConfig,
    AWQConfig,
    TEQConfig,
    StaticQuantConfig,
    SmoothQuantConfig,
    HQQConfig,
    convert,
    get_default_AutoRound_config,
    prepare
)
from intel_extension_for_transformers.tools.utils import (
    is_ipex_available,
    is_autoround_available,
)
from transformers import AutoTokenizer

if is_ipex_available():
    import intel_extension_for_pytorch as ipex

if is_autoround_available():
    from auto_round.export.export_to_itrex.model_wrapper import WeightOnlyLinear as auto_round_woqlinear # pylint: disable=E0401

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


def unpack_weight(qweight, scales, qzeros, q_config):
    sym = q_config.sym
    bits = q_config.bits
    wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0)

    zeros = torch.bitwise_right_shift(
        torch.unsqueeze(qzeros, 2).expand(-1, -1, 32 // bits), wf.unsqueeze(0)
    ).to(torch.int16 if bits == 8 else torch.int8)
    torch.bitwise_and(zeros, (2**bits) - 1, out=zeros)
    if bits == 8:
        zeros = zeros.to(torch.int8 if sym else torch.uint8)
    # due to INC minus one
    zeros = zeros + 1
    try:
        zeros = zeros.reshape(scales.shape)
    except:
        # zeros and scales have different iteam numbers.
        # remove 1 (due to 0 + 1 in line 68)
        zeros = zeros[zeros != 1]
        zeros = zeros.reshape(scales.shape)

    # due to INC asym return torch.uint8 but backend request int8,
    # change it to int8 with offset 128
    if not sym and bits == 8:
        zeros = (zeros.to(torch.int32) - 128).to(torch.int8)

    weight = torch.bitwise_right_shift(
        torch.unsqueeze(qweight, 1).expand(-1, 32 // bits, -1), wf.unsqueeze(-1)
    ).to(torch.int16 if bits == 8 else torch.int8)
    torch.bitwise_and(weight, (2**bits) - 1, out=weight)
    if bits == 8:
        # due to INC add shift bias for sym
        if sym:
            shift_bias = 2 ** (bits - 1)
            weight -= shift_bias
        weight = weight.to(torch.int8 if sym else torch.uint8)
        # due to INC asym return torch.uint8 but backend request int8,
        # change it to int8 with offset 128
        if not sym:
            weight = (weight.to(torch.int32) - 128).to(torch.int8)
    return weight, scales, zeros


def replace_linear(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    device="cpu",
    empty_weights=False,
):
    if modules_to_not_convert is None:
        # output_layer is chatglm last layer name
        # embed_out is dolly_v2 last layer name
        modules_to_not_convert = ["lm_head", "output_layer", "embed_out"]
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
    """Private method that wraps the recursion for module replacement.

    Returns the converted model and a boolean that indicates if the conversion has been successfully or not.
    """
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)
        is_removed = False
        use_optimum_format = getattr(module, "use_optimum_format", False) or \
            quantization_config.weight_dtype not in [
                "fp8_e5m2",
                "fp8_e4m3",
                "fp4",
                "nf4",
                "int4_fullrange",
            ]

        if (
            isinstance(module, torch.nn.Linear)
            or isinstance(module, WeightOnlyLinear)
            or (is_autoround_available() and isinstance(module, auto_round_woqlinear))
            or (
                is_ipex_available()
                and isinstance(module, ipex.nn.utils._weight_prepack._IPEXLinear)
            )
        ) and (name not in modules_to_not_convert):
            # Check if the current key is not in the `modules_to_not_convert`
            if not any(
                key in ".".join(current_key_name) for key in modules_to_not_convert
            ):
                with init_empty_weights():
                    in_features = module.in_features
                    out_features = module.out_features
                    if (
                        device == "cpu"
                        or device == torch.device("cpu")
                        or device == "auto"
                    ):
                        from .nn.modules import (
                            QuantizedLinearQBits,
                        )  # TODO: QuantizedLinearINT4, QuantizedLinearINT8

                        use_optimum_format = getattr(module, "use_optimum_format", False) or \
                            quantization_config.weight_dtype not in [
                                "fp8_e5m2",
                                "fp8_e4m3",
                                "fp4",
                                "nf4",
                                "int4_fullrange",
                            ]

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
                            compression_dtype=getattr(module, "compression_dtype", torch.int32),
                            compression_dim=getattr(module, "compression_dim", 1),
                            device=device,
                            use_optimum_format=use_optimum_format,
                        )
                    elif device == "xpu" or device == torch.device("xpu"):
                        from intel_extension_for_pytorch.nn.utils._quantize_convert \
                            import WeightOnlyQuantizedLinear as ipex_linear  # pylint: disable=E0401
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
                            compression_dtype=getattr(module, "compression_dtype", torch.int8),
                            compression_dim=getattr(module, "compression_dim", 0),
                            device=device,
                            use_optimum_format=getattr(module, "use_optimum_format", False),
                        )
                        if quantization_config.quant_method.value == "gptq":
                            g_idx = getattr(module, "g_idx", torch.zeros(in_features, dtype=torch.int32).to(device))
                        else:
                            g_idx = None
                        model._modules[name].set_scales_zps_gidx(
                            (
                                module.scales
                                if hasattr(module, "scales")
                                else torch.ones(
                                    (
                                        out_features,
                                        math.ceil(
                                            in_features / quantization_config.group_size
                                        ),
                                    ),
                                    dtype=convert_dtype_str2torch(
                                        quantization_config.compute_dtype
                                    ),
                                    device=torch.device(device),
                                )
                            ),
                            module.qzeros if hasattr(module, "qzeros") else None,
                            g_idx,
                        )
                    else:
                        raise Exception(
                            "{} device Unsupported weight only quantization!".format(
                                device
                            )
                        )

                    is_replaced = True
                    # Store the module class in case we need to transpose the weight later
                    model._modules[name].source_cls = type(module)
                    # Force requires grad to False to avoid unexpected errors
                    model._modules[name].requires_grad_(False)
                if device == "cpu" or device == torch.device("cpu") or device == "auto":
                    if quantization_config.weight_dtype in [
                        "fp8_e5m2",
                        "fp8_e4m3",
                        "nf4",
                        "fp4",
                        "int4_fullrange",
                    ]:
                        model._modules[name].set_fp_weights_bias(
                            module.weight.data,
                            None if module.bias is None else module.bias.data,
                        )
                    else:
                        int_weight, scales, zeros = unpack_weight(
                            module.qweight,
                            module.scales,
                            module.qzeros,
                            quantization_config,
                        )
                        int_weight = int_weight.view(-1, int_weight.shape[-1])

                        model._modules[name].set_weights_bias(
                            int_weight,
                            scales,
                            zeros,
                            module.g_idx if hasattr(module, "g_idx") else None,
                            quantization_config,
                            bias=None if module.bias is None else module.bias.data,
                        )
                else:
                    if not hasattr(module, "qweight"):
                        n_pack = (
                            8 // DTYPE_BITS_MAPPING[quantization_config.weight_dtype]
                        )
                        weight = torch.zeros(
                            (math.ceil(out_features / n_pack), in_features),
                            dtype=torch.int8,
                            device=torch.device(device),
                        )
                    model._modules[name].set_weights_bias(
                        module.qweight.data if hasattr(module, "qweight") else weight,
                        None if module.bias is None else module.bias.data,
                    )
                    del module
                    gc.collect()
                    is_removed = True

        if not is_removed and len(list(module.children())) > 0:  # pylint: disable=E1101
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

        assert (
            hasattr(torch, "xpu") and torch.xpu.is_available()
        ), "There is no xpu device in this system!"
    calib_iters = config.calib_iters
    calib_dataset = config.dataset
    model_device = next(model.parameters()).device

    if (
        calib_dataloader is None
        and config.quant_method.value not in ["rtn"]
        and calib_dataset is not None
    ):
        from datasets import load_dataset
        from torch.utils.data import DataLoader

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
            elif "code" in examples:
                example = config.tokenizer(examples["code"])
            elif "text" in examples:
                example = config.tokenizer(examples["text"])
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
                    input_ids[:512]
                    if (len(input_ids) > 512 and config.quant_method.value != "gptq")
                    else input_ids
                )
                input_ids_padded.append(input_ids)
            return torch.vstack(input_ids_padded)

        def collate_batch_for_autoround(batch):
            input_ids_padded = []
            for text in batch:
                input_ids = text["input_ids"]
                if input_ids.shape[0] < config.calib_len:
                    continue
                input_ids = input_ids[: config.calib_len]
                input_ids_list = input_ids.tolist()
                if input_ids_list.count(input_ids_list[-1]) > config.calib_len // 2:
                    continue
                input_ids_padded.append(input_ids)
            if len(input_ids_padded) == 0:
                return None

            return torch.vstack(input_ids_padded)


        calib_dataloader = DataLoader(
            tokenized_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_batch,
        )
    if calib_func is None and config.quant_method.value == "awq":

        def default_calib_func(model):
            """This is the default calibration function, the dataset is NeelNanda/pile-10k,
            the default calib_iters is 100."""
            for i, (input_ids) in enumerate(calib_dataloader):
                if i >= calib_iters:
                    break
                model(
                    input_ids=input_ids,
                )

        calib_func = default_calib_func
        logger.info(
            "The default calibration function is used, "
            + "the calibration dataset is NeelNanda/pile-10k,"
            + "batchsize is 1 and calibration iteration is 100."
        )
    orig_dtype = torch.float32
    for param in model.parameters():
        orig_dtype = param.dtype
        if orig_dtype != torch.float32:
            model.to(dtype=torch.float32)
        break
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
        # mapping to INC config
        if config.quant_method.value == "rtn":
            quant_config = RTNConfig(

            )
        elif config.quant_method.value == "hqq":
            quant_config = HQQConfig(

            )
        elif config.quant_method.value == "awq":
            quant_config = AWQConfig(

            ) 
        elif config.quant_method.value == "teq":
            quant_config = TEQConfig(

            )
        elif config.quant_method.value == "gptq":
            quant_config = GPTQConfig(

            )
        elif config.quant_method.value == "autoround":
            quant_config = AutoRoundConfig(
                dtype=config.dtype,
                bits=config.bits,
                use_sym=config.use_sym,
                group_size=config.group_size,
                enable_full_range=config.enable_full_range,
                batch_size=config.batch_size,
                lr_scheduler=config.lr_scheduler,
                use_quant_input=config.use_quant_input,
                enable_minmax_tuning=config.enable_minmax_tuning,
                lr=config.lr,
                minmax_lr=config.minmax_lr,
                low_gpu_mem_usage=config.low_gpu_mem_usage,
                iters=config.iters,
                seqlen=config.seq_len,
                n_samples=config.n_samples,
                sampler=config.sampler,
                seed=config.seed,
                n_blocks=config.n_blocks,
                gradient_accumulate_steps=config.gradient_accumulate_steps,
                not_use_best_mse=config.not_use_best_mse,
                dynamic_max_gap=config.dynamic_max_gap,
                scale_dtype=config.scale_dtype,
                white_list=config.white_list,
            )
            quant_config.set_local(".*lm_head", AutoRoundConfig(w_dtype="fp32", act_dtype="fp32"))
            quant_config.set_local(".*output_layer", AutoRoundConfig(w_dtype="fp32", act_dtype="fp32"))
            quant_config.set_local(".*embed_out", AutoRoundConfig(w_dtype="fp32", act_dtype="fp32"))
            logger.info(f"Do AutoRound with config {quant_config}")

            run_fn = get_autoround_default_run_fn
            run_args = (
                tokenizer,
                dataset,
                quant_config.n_samples,
                quant_config.seq_len,
                quant_config.seed,
                quant_config.batch_size,
                "train"
            )
            inc_model = prepare(model=model, quant_config=quant_config)
            run_fn(model, *run_args)
            inc_model = convert(inc_model)
            inc_model.eval()

        else:
            assert False, "The Supported algorithm are RTN, AWQ, TEQ, GPTQ, AUTOROUND"

        if device == "xpu" or device == torch.device("xpu"):
            model = inc_model.export_compressed_model(
                compression_dtype=torch.int8,
                compression_dim=0,
                use_optimum_format=False,
                scale_dtype=convert_dtype_str2torch(config.scale_dtype),
                device="xpu",
            )

            q_model = replace_linear(model, None, None, config, device=device)
        else:
            if config.weight_dtype not in ["nf4", "fp4", "int4_fullrange"]:
                inc_model = inc_model.export_compressed_model(use_optimum_format=True)
                inc_model.eval()
                q_model = replace_linear(inc_model, None, None, config, device=device)
            else:
                q_model = replace_linear(
                    inc_model.model, None, None, config, device=device
                )

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
        assert False, "Unsupported {} for quantize weight only by IPEX backend".format(
            config.weight_dtype
        )
    return bits
