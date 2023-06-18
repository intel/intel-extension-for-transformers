import argparse
import math
import os
import numpy as np
import torch
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
    parser.add_argument(
        "-c",
        "--captions",
        type=str,
        nargs="+",
        default=["Snow mountain of sunrise."],
        help="Text used to generate images.",
    )
    parser.add_argument(
        "-n",
        "--images_num",
        type=int,
        default=4,
        help="How much images to generate.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Seed for random process.",
    )
    parser.add_argument(
        "-ci",
        "--cuda_id",
        type=int,
        default=0,
        help="cuda_id.",
    )
    args = parser.parse_args()
    return args


def image_grid(imgs, rows, cols):
    if not len(imgs) == rows * cols:
        raise ValueError("The specified number of rows and columns are not correct.")

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def generate_images(
    pipeline,
    prompt="Snow mountain of sunrise.",
    guidance_scale=7.5,
    num_inference_steps=50,
    num_images_per_prompt=1,
    seed=42,
):
    generator = torch.Generator(pipeline.vae.device).manual_seed(seed)
    images = pipeline(
        prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        num_images_per_prompt=num_images_per_prompt,
    ).images
    _rows = int(math.sqrt(num_images_per_prompt))
    grid = image_grid(images, rows=_rows, cols=num_images_per_prompt // _rows)
    return grid, images


args = parse_args()
# Load models and create wrapper for stable diffusion
#unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

# pipeline = StableDiffusionPipeline.from_pretrained(
#     args.pretrained_model_name_or_path, unet=unet
# )
device = 'cpu'
pipeline = StableDiffusionPipeline.from_pretrained(
     args.pretrained_model_name_or_path
).to(device)
unet = pipeline.unet
pipeline.safety_checker = lambda images, clip_input: (images, False)

import copy
unet_fp32 = copy.deepcopy(unet)
unet_fp32.eval()
if os.path.exists(os.path.join(args.pretrained_model_name_or_path, "best_model.pt")):
    unet = load(args.pretrained_model_name_or_path, model=unet)
    unet.eval()
    setattr(pipeline, "unet", unet)
elif os.path.exists(os.path.join('./', "fake_quant_model_qinit.pt")):
#elif os.path.exists(os.path.join(args.pretrained_model_name_or_path, "fake_quant_model_qinit.pt")):
    from fake_quant_modules import find_and_replace, convert2quantized_model, disable_all_observers
    find_and_replace(unet)
    # disable_all_observers(unet)
    unet.load_state_dict(torch.load(os.path.join('./', "fake_quant_model_qinit.pt")))
    unet = convert2quantized_model(unet)
    unet.eval()
    setattr(pipeline, "unet", unet)
else:
    if torch.cuda.is_available():
        unet = unet.to(torch.device("cuda", args.cuda_id))

#import pdb;pdb.set_trace()
###############################################################

# pipeline = pipeline.to(unet.device)
dtype = torch.float32
export2onnx = True
if export2onnx:
    import onnx
    import onnxruntime as ort
    from neural_compressor.config import Torch2ONNXConfig
    from neural_compressor.model import Model
    onnx_model_path = 'onnx_fp32/model.onnx'
    os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)
    if os.path.exists(os.path.dirname(onnx_model_path)):
        inc_model = Model(unet_fp32)
        fp32_onnx_config = Torch2ONNXConfig(
            dtype="fp32",
            example_inputs={'sample':torch.randn(2, 4, 64, 64), 'timestep':torch.randn(2).to(device=device), 'encoder_hidden_states':torch.randn(2, 77, 768)},
            input_names=['sample', 'timestep', 'encoder_hidden_states'],
            output_names=['output'],
            dynamic_axes={"sample": {0: "batch_size"},
                        "timestep": {0: "batch_size"},
                        "encoder_hidden_states": {0: "batch_size", 1:"seq_length"},
                        "output": {0: "batch_size"}},
            do_constant_folding=True,
        )
        inc_model.export(onnx_model_path, fp32_onnx_config)

        # torch.onnx.export(
        #     unet_fp32,
        #     #pipeline.unet,
        #     args=(
        #         torch.randn(2, 4, 64, 64).to(device=device,dtype=dtype),
        #         torch.randn(2).to(device=device, dtype=dtype),
        #         torch.randn(2, 77, 768).to(device=device, dtype=dtype),
        #         #False,
        #     ),
        #     f=onnx_model_path,
        #     input_names=["sample", "timestep", "encoder_hidden_states"],
        #     #input_names=["sample", "timestep", "encoder_hidden_states", "return_dict"],
        #     output_names=["out_sample"],  # has to be different from "sample" for correct tracing
        #     dynamic_axes={
        #         "sample": {
        #             0: "batch",
        #             1: "channels",
        #             2: "height",
        #             3: "width"
        #         },
        #         "timestep": {
        #             0: "batch"
        #         },
        #         "encoder_hidden_states": {
        #             0: "batch",
        #             1: "sequence"
        #         },
        #     },
        #     do_constant_folding=True,
        #     opset_version=14,
        # )

        unet_dir = os.path.dirname(onnx_model_path)
        unet_fp32_onnx = onnx.load(onnx_model_path)
        # clean up existing tensor files
        import shutil, shlex
        shutil.rmtree(unet_dir)
        os.mkdir(shlex.quote(unet_dir))
        # collate external tensor files into one
        onnx.save_model(
            unet_fp32_onnx,
            onnx_model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="weights.pb",
            convert_attribute=False,
        )

        # sess_options = ort.SessionOptions()
        # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        # sess_options.optimized_model_filepath = onnx_model_path
        # ort.InferenceSession(onnx_model_path, sess_options)

    onnx_model_path = 'onnx_int8/model.onnx'
    os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)
    if os.path.exists(os.path.dirname(onnx_model_path)):
        from neural_compressor.config import Torch2ONNXConfig
        from neural_compressor.model import Model
        # int8_onnx_config = Torch2ONNXConfig(
        #     dtype="int8",
        #     opset_version=14,
        #     quant_format="QDQ", # or QDQ
        #     example_inputs={'sample':torch.randn(2, 4, 64, 64), 'timestep':torch.randint(0, 1000, (2,)).long(), 'encoder_hidden_states':torch.randn(2, 4, 768)},
        #     input_names=['sample', 'timestep', 'encoder_hidden_states'],
        #     output_names=['output'],
        #     dynamic_axes={"sample": {0: "batch_size"},
        #                 "timestep": {0: "batch_size"},
        #                 "encoder_hidden_states": {0: "batch_size", 1:"seq_length"},
        #                 "output": {0: "batch_size"}},
        # )
        # q_model = Model(unet)
        # q_model.q_config = {'approach':'quant_aware_training', 'reduce_range':False}
        #q_model.export(onnx_model_path, int8_onnx_config)

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
        
        dtype = torch.float32
        torch.onnx.export(
            unet,
            args=(
                torch.randn(2, 4, 64, 64).to(device=device,dtype=dtype),
                torch.randn(2).to(device=device, dtype=dtype),
                torch.randn(2, 77, 768).to(device=device, dtype=dtype),
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


        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        sess_options.optimized_model_filepath = onnx_model_path
        ort.InferenceSession(onnx_model_path, sess_options)

    sess = ort.InferenceSession(onnx_model_path, providers=ort.get_available_providers())

    class Onnx_UNet(torch.nn.Module):
        def __init__(self, session) -> None:
            super().__init__()
            self.model = session
            self.config = unet_fp32.config
            self.device = unet_fp32.device
            self.in_channels = unet_fp32.in_channels

        def forward(self, sample, timestep, encoder_hidden_states,
                    class_labels=None,
                    timestep_cond=None,
                    attention_mask=None,
                    cross_attention_kwargs=None,
                    down_block_additional_residuals=None,
                    mid_block_additional_residual=None,):
            output = sess.run(None, {'sample':np.array(sample), 'timestep':np.array([timestep]), 'encoder_hidden_states':np.array(encoder_hidden_states)})
            # import pdb; pdb.set_trace()
            return UNet2DConditionOutput(sample=torch.tensor(output[0]).to(pipeline.device))

    unet = Onnx_UNet(sess)
    setattr(pipeline, "unet", unet)

for caption in args.captions:
    print(caption)
    os.makedirs(os.path.join(args.pretrained_model_name_or_path, "results"), exist_ok=True)
    grid, images = generate_images(pipeline, prompt=caption, num_images_per_prompt=args.images_num, seed=args.seed)
    grid.save(os.path.join(args.pretrained_model_name_or_path, "results", "{}.png".format("_".join(caption.split()))))
    dirname = os.path.join(args.pretrained_model_name_or_path, "results", "_".join(caption.split()))
    os.makedirs(dirname, exist_ok=True)
    for idx, image in enumerate(images):
        image.save(os.path.join(dirname, "{}.png".format(idx + 1)))

