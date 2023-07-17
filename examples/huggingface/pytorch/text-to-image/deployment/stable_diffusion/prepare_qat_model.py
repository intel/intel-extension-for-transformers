import argparse
import math
import os
import numpy as np
import torch
import onnx
import copy
from neural_compressor.utils.pytorch import load
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.models.unet_2d_condition import UNet2DConditionOutput


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    args = parser.parse_args()
    return args


args = parse_args()
# Load models and create wrapper for stable diffusion
#unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

device = 'cpu'
pipeline = StableDiffusionPipeline.from_pretrained(
     args.pretrained_model_name_or_path
).to(device)
unet = pipeline.unet
#pipeline.safety_checker = lambda images, clip_input: (images, False)

# import copy
# unet_fp32 = copy.deepcopy(unet)
# unet_fp32.eval()
# if os.path.exists(os.path.join(args.pretrained_model_name_or_path, "best_model.pt")):
#     unet = load(args.pretrained_model_name_or_path, model=unet)
#     unet.eval()
#     setattr(pipeline, "unet", unet)
# elif os.path.exists(os.path.join('./', "fake_quant_model_qinit.pt")):
#elif os.path.exists(os.path.join(args.pretrained_model_name_or_path, "fake_quant_model_qinit.pt")):
from fake_quant_modules import find_and_replace, convert2quantized_model, disable_all_observers
find_and_replace(unet)
# disable_all_observers(unet)
unet.load_state_dict(torch.load(os.path.join('./', "fake_quant_model_qinit.pt")))
unet = convert2quantized_model(unet)
unet.eval()
setattr(pipeline, "unet", unet)

# pipeline = pipeline.to(unet.device)

# onnx_model_path = 'onnx_fp32/model.onnx'
# os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)
# if os.path.exists(os.path.dirname(onnx_model_path)):
#     torch.onnx.export(
#         unet_fp32,
#         #pipeline.unet,
#         args=(
#             torch.randn(2, 4, 64, 64).to(device=device,dtype=torch.float32),
#             torch.randn(2).to(device=device, dtype=torch.float32),
#             torch.randn(2, 77, 768).to(device=device, dtype=torch.float32),
#             #False,
#         ),
#         f=onnx_model_path,
#         input_names=["sample", "timestep", "encoder_hidden_states"],
#         output_names=["out_sample"],  # has to be different from "sample" for correct tracing
#         dynamic_axes={
#             "sample": {
#                 0: "batch",
#                 1: "channels",
#                 2: "height",
#                 3: "width"
#             },
#             "timestep": {
#                 0: "batch"
#             },
#             "encoder_hidden_states": {
#                 0: "batch",
#                 1: "sequence"
#             },
#         },
#         do_constant_folding=True,
#         opset_version=14,
#     )

#     unet_dir = os.path.dirname(onnx_model_path)
#     unet_fp32_onnx = onnx.load(onnx_model_path)
#     # clean up existing tensor files
#     import shutil, shlex
#     shutil.rmtree(unet_dir)
#     os.mkdir(shlex.quote(unet_dir))
#     # collate external tensor files into one
#     onnx.save_model(
#         unet_fp32_onnx,
#         onnx_model_path,
#         save_as_external_data=True,
#         all_tensors_to_one_file=True,
#         location="weights.pb",
#         convert_attribute=False,
#     )




onnx_model_path = 'onnx_int8/model.onnx'
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
            #False,
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