import argparse
import os
import shutil
from pathlib import Path

import torch
import onnx
from diffusers import StableDiffusionPipeline


@torch.no_grad()

def _export_bf16_onnx_model(fp32_model_path):
    #fp32_model_path = fp32_model_path.as_posix()
    bf16_path = ''
    for i, path in enumerate(fp32_model_path.split('/')):
        if i == len(fp32_model_path.split('/')) - 1:
            bf16_path += 'bf16-' + path
        else:
            bf16_path += path+'/'
    # if 'unet' in fp32_model_path:
    #     import pdb; pdb.set_trace()
    model = onnx.load(fp32_model_path)
    bf16_type_list = ['MatMul', 'Gemm', 'Conv']
    bf16_tensor_name_list = []
    for node in model.graph.node:
        if node.op_type in bf16_type_list:
            for inp in node.input:
                bf16_tensor_name_list.append(inp)
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper
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
    onnx.save(model, bf16_path)
    # os.remove(fp32_model_path)
    print('*********bf16 onnx model exported!**************')
    print(bf16_path)
    print('*********bf16 onnx model exported!**************')


def prepare_model(model_name: str, output_path: Path, opset: int, bf16: bool = False):
    device = 'cpu'
    dtype = torch.float32
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

    output_path = Path(output_path)
    text_encoder = output_path / "text_encoder" / "model.onnx"
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

    if bf16 == True:
        _export_bf16_onnx_model(text_encoder.as_posix())

    del pipeline.text_encoder

    # UNET
    unet_in_channels = pipeline.unet.config.in_channels
    unet_sample_size = pipeline.unet.config.sample_size
    unet_path = output_path / "unet" / "model.onnx"
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
    unet_dir = os.path.dirname(unet_model_path)
    unet = onnx.load(unet_model_path)
    # clean up existing tensor files
    shutil.rmtree(unet_dir)
    os.mkdir(unet_dir)
    # collate external tensor files into one
    onnx.save_model(
        unet,
        unet_model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.pb",
        convert_attribute=False,
    )
    if bf16 == True:
        _export_bf16_onnx_model(unet_path.as_posix())
        fp32_path = unet_model_path
        bf16_path = ''
        for i, path in enumerate(fp32_path.split('/')):
            if i == len(fp32_path.split('/')) - 1:
                bf16_path += 'bf16-' + path
            else:
                bf16_path += path+'/'
        unet_bf16 = onnx.load(bf16_path)
        onnx.save_model(
            unet_bf16,
            bf16_path,
            location="weights_bf16.pb",
        )
    del pipeline.unet

    # VAE DECODER
    vae_decoder = pipeline.vae
    vae_latent_channels = vae_decoder.config.latent_channels
    # forward only through the decoder part
    vae_decoder.forward = pipeline.vae.decode

    vae_decoder_path = output_path / "vae_decoder" / "model.onnx"
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

    if bf16 == True:
        _export_bf16_onnx_model(vae_decoder_path.as_posix())
    del pipeline.vae


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
    parser.add_argument("--bf16",
                        default=False,
                        help="Export the models in `float16` mode")

    args = parser.parse_args()

    prepare_model(args.input_model, args.output_path, args.opset, args.bf16)