import argparse
import math
import os
import shlex
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

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
        "-qm",
        "--quantized_model_name_or_path",
        type=str,
        default=None,
        help="Path to quantized unet model.",
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
    with torch.no_grad():
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

if __name__ == "__main__":
    args = parse_args()
    # Load models and create wrapper for stable diffusion
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, unet=unet
    )
    pipeline.safety_checker = lambda images, clip_input: (images, False)
    results_path = args.pretrained_model_name_or_path

    if args.quantized_model_name_or_path and os.path.exists(args.quantized_model_name_or_path):
        from quantization_modules import load_int8_model
        unet = load_int8_model(unet, args.quantized_model_name_or_path)
        unet.eval()
        setattr(pipeline, "unet", unet)
        results_path = os.path.dirname(args.quantized_model_name_or_path)
    else:
        if torch.cuda.is_available():
            unet = unet.to(torch.device("cuda", args.cuda_id))

    pipeline = pipeline.to(unet.device)
    for caption in args.captions:
        print(caption)
        os.makedirs(os.path.join(results_path, "results"), exist_ok=True)
        grid, images = generate_images(pipeline, prompt=caption, num_images_per_prompt=args.images_num, seed=args.seed)
        grid.save(os.path.join(results_path, "results", "{}.png".format("_".join(caption.split()))))
        dirname = os.path.join(results_path, "results", "_".join(caption.split()))
        os.makedirs(shlex.quote(dirname), exist_ok=True)
        for idx, image in enumerate(images):
            image.save(os.path.join(dirname, "{}.png".format(idx + 1)))
