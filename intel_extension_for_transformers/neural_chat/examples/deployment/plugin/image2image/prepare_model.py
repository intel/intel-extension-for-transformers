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

import argparse
import os
import shutil
from pathlib import Path

import torch
import onnx
from diffusers import StableDiffusionPipeline
import shlex


@torch.no_grad()
def _export_bf16_onnx_model(fp32_model_path, bf16_model_path):
    model = onnx.load(fp32_model_path)
    bf16_type_list = ['MatMul', 'Gemm', 'Conv']
    bf16_tensor_name_list = []
    for node in model.graph.node:
        if node.op_type in bf16_type_list:
            for inp in node.input:
                bf16_tensor_name_list.append(inp)
    import numpy as np
    from onnx import TensorProto, numpy_helper
    for tensor in model.graph.initializer:
        if tensor.name in bf16_tensor_name_list:

            def fp32_to_bf16(fp32_np):
                assert (fp32_np.dtype == np.float32)
                int32_np = fp32_np.view(dtype=np.int32)
                int32_np = int32_np >> 16
                bf16_np = int32_np.astype(np.int16)
                return bf16_np

            fp16_data = fp32_to_bf16(numpy_helper.to_array(tensor))
            tensor.raw_data = fp16_data.tobytes()
            tensor.data_type = TensorProto.BFLOAT16
    onnx.save(model, bf16_model_path)


def prepare_model(
    model_name: str,
    output_path: Path,
    opset: int,
    expected_dtype: str,
    fake_quant_model_qinit_path: str,
    fake_quant_model_qinit_name: str
):
    device = 'cpu'
    dtype = torch.float32
    output_path = Path(output_path)
    pipeline = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=dtype).to(device)

    # TEXT ENCODER
    num_tokens = pipeline.text_encoder.config.max_position_embeddings
    text_hidden_size = pipeline.text_encoder.config.hidden_size
    text_input = pipeline.tokenizer(
        "A sample prompt",
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    text_encoder = output_path / "text_encoder_fp32" / "model.onnx"
    text_encoder.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        pipeline.text_encoder,
        args=(text_input.input_ids.to(device=device, dtype=torch.int32)),
        f=text_encoder.as_posix(),
        input_names=["input_ids"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            "input_ids": {
                0: "batch",
                1: "sequence"
            },
        },
        do_constant_folding=True,
        opset_version=opset,
    )

    if expected_dtype == 'bf16' or expected_dtype == 'qat_int8':
        text_encoder_bf16 = output_path / "text_encoder_bf16" / "model.onnx"
        text_encoder_bf16_dir = output_path / "text_encoder_bf16"
        if os.path.exists(text_encoder_bf16_dir):
            shutil.rmtree(text_encoder_bf16_dir)
        os.mkdir(shlex.quote(text_encoder_bf16_dir.as_posix()))
        _export_bf16_onnx_model(text_encoder.as_posix(), text_encoder_bf16.as_posix())

    del pipeline.text_encoder

    # UNET
    if expected_dtype == 'qat_int8':
        prepare_qat_model(model_name, output_path, fake_quant_model_qinit_path, fake_quant_model_qinit_name)

    unet_in_channels = pipeline.unet.config.in_channels
    unet_sample_size = pipeline.unet.config.sample_size
    unet_path = output_path / "unet_fp32" / "model.onnx"
    unet_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        pipeline.unet,
        args=(
            torch.randn(2, unet_in_channels, unet_sample_size, unet_sample_size).to(device=device,
                                                                                    dtype=dtype),
            torch.randn(2).to(device=device, dtype=dtype),
            torch.randn(2, num_tokens, text_hidden_size).to(device=device, dtype=dtype),
            False,
        ),
        f=unet_path.as_posix(),
        input_names=["sample", "timestep", "encoder_hidden_states", "return_dict"],
        output_names=["out_sample"],  # has to be different from "sample" for correct tracing
        dynamic_axes={
            "sample": {
                0: "batch",
                1: "channels",
                2: "height",
                3: "width"
            },
            "timestep": {
                0: "batch"
            },
            "encoder_hidden_states": {
                0: "batch",
                1: "sequence"
            },
        },
        do_constant_folding=True,
        opset_version=opset,
    )

    unet_model_path = str(unet_path.absolute().as_posix())

    if expected_dtype == 'bf16' or expected_dtype == 'qat_int8':
        unet_bf16_model_path = output_path / "unet_bf16" / "model.onnx"
        unet_bf16_dir = output_path / "unet_bf16"
        if os.path.exists(unet_bf16_dir):
            shutil.rmtree(unet_bf16_dir)
        os.mkdir(shlex.quote(unet_bf16_dir.as_posix()))
        _export_bf16_onnx_model(unet_path.as_posix(), unet_bf16_model_path.as_posix())
        unet_bf16_model = onnx.load(unet_bf16_model_path)

    unet_dir = os.path.dirname(unet_model_path)
    unet = onnx.load(unet_model_path)
    # clean up existing tensor files
    shutil.rmtree(unet_dir)
    os.mkdir(shlex.quote(unet_dir))
    # collate external tensor files into one
    onnx.save_model(
        unet,
        unet_model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.pb",
        convert_attribute=False,
    )
    if expected_dtype == 'bf16' or expected_dtype == 'qat_int8':
        unet_bf16_model_path = str(unet_bf16_model_path.absolute().as_posix())
        onnx.save_model(
            unet_bf16_model,
            unet_bf16_model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="weights.pb",
        )
    del pipeline.unet

    # VAE DECODER
    vae_decoder = pipeline.vae
    vae_latent_channels = vae_decoder.config.latent_channels
    # forward only through the decoder part
    vae_decoder.forward = pipeline.vae.decode

    vae_decoder_path = output_path / "vae_decoder_fp32" / "model.onnx"
    vae_decoder_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        vae_decoder,
        args=(
            torch.randn(1, vae_latent_channels, unet_sample_size, unet_sample_size).to(device=device,
                                                                                       dtype=dtype),
            False,
        ),
        f=vae_decoder_path.as_posix(),
        input_names=["latent_sample", "return_dict"],
        output_names=["sample"],
        dynamic_axes={
            "latent_sample": {
                0: "batch",
                1: "channels",
                2: "height",
                3: "width"
            },
        },
        do_constant_folding=True,
        opset_version=opset,
    )

    if expected_dtype == 'bf16' or expected_dtype == 'qat_int8':
        vae_decoder_bf16_model = output_path / "vae_decoder_bf16" / "model.onnx"
        vae_decoder_bf16_dir = output_path / "vae_decoder_bf16"
        if os.path.exists(vae_decoder_bf16_dir):
            shutil.rmtree(vae_decoder_bf16_dir)
        os.mkdir(shlex.quote(vae_decoder_bf16_dir.as_posix()))
        _export_bf16_onnx_model(vae_decoder_path.as_posix(), vae_decoder_bf16_model.as_posix())
    del pipeline.vae


def prepare_qat_model(
    model_name: str,
    output_path: Path,
    fake_quant_model_qinit_path: str = "./",
    fake_quant_model_qinit_name: str = "fake_quant_model_qinit.pt"
):
    device = 'cpu'
    output_path = Path(output_path)
    pipeline = StableDiffusionPipeline.from_pretrained(model_name).to(device)
    unet = pipeline.unet

    from quantization_modules import find_and_replace, convert2quantized_model
    find_and_replace(unet)
    unet.load_state_dict(torch.load(os.path.join(fake_quant_model_qinit_path, fake_quant_model_qinit_name)))
    unet = convert2quantized_model(unet)
    unet.eval()
    setattr(pipeline, "unet", unet)

    onnx_model_path = output_path / "unet_qat_int8" / "model.onnx"
    os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)
    if os.path.exists(os.path.dirname(onnx_model_path)):
        def model_wrapper(model_fn):
        # export doesn't support a dictionary output, so manually turn it into a tuple
        # refer to https://discuss.tvm.apache.org/t/how-to-deal-with-prim-dictconstruct/11978
            def wrapper(*args, **kwargs):
                output = model_fn(*args, **kwargs)
                if isinstance(output, dict):
                    return tuple(v for v in output.values() if v is not None)
                else:
                    return output
            return wrapper
        unet.forward = model_wrapper(unet.forward)

        torch.onnx.export(
            unet,
            args=(
                torch.randn(2, 4, 64, 64).to(device=device,dtype=torch.float32),
                torch.randn(2).to(device=device, dtype=torch.float32),
                torch.randn(2, 77, 768).to(device=device, dtype=torch.float32),
            ),
            f=onnx_model_path,
            input_names=["sample", "timestep", "encoder_hidden_states"],# "return_dict"],
            output_names=["out_sample"],  # has to be different from "sample" for correct tracing
            dynamic_axes={
                "sample": {
                    0: "batch",
                    1: "channels",
                    2: "height",
                    3: "width"
                },
                "timestep": {
                    0: "batch"
                },
                "encoder_hidden_states": {
                    0: "batch",
                    1: "sequence"
                },
            },
            do_constant_folding=True,
            opset_version=14,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_model",
        type=str,
        required=True,
        help="Path to the `diffusers` checkpoint to convert (either a local directory or on the Hub).",
    )
    parser.add_argument('--pattern_config',
                        default="./pattern_config",
                        type=str,
                        help="The fusion pattern config path for the nerual engine.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output model.")
    parser.add_argument(
        "--opset",
        default=14,
        type=int,
        help="The version of the ONNX operator set to use.",
    )
    parser.add_argument("--bf16", action="store_true", help="Export the models in `bfloat16` mode")
    parser.add_argument("--qat_int8", action="store_true", help="Export the models in `bfloat16` mode")
    parser.add_argument(
        "--fake_quant_model_qinit_path",
        type=str,
        default="./",
        help="Path to the fake_quant_model_qinit",
    )
    parser.add_argument(
        "--fake_quant_model_qinit_name",
        type=str,
        default="fake_quant_model_qinit.pt",
        help="Name of the fake_quant_model_qinit",
    )

    args = parser.parse_args()

    expected_dtype = 'fp32'
    if args.bf16:
        expected_dtype = 'bf16'
    elif args.qat_int8:
        expected_dtype = 'qat_int8'

    prepare_model(
        args.input_model,
        args.output_path,
        args.opset,
        expected_dtype,
        args.fake_quant_model_qinit_path,
        args.fake_quant_model_qinit_name
    )
