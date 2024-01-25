#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import copy
import logging
import os
import time
import numpy as np
import pathlib

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.datasets as dset
import torchvision.transforms as transforms
from text2images import StableDiffusionPipelineMixedPrecision, StableDiffusionXLPipelineMixedPrecision

logging.getLogger().setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="", help="Model path")
    parser.add_argument("--int8_model_path", type=str, default="", help="INT8 model path")
    parser.add_argument("--dataset_path", type=str, default="", help="COCO2017 dataset path")
    parser.add_argument("--output_dir", type=str, default=None,help="output path")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument('--precision', type=str, default="fp32", help='precision: fp32, bf16, int8, int8-bf16')
    parser.add_argument('-i', '--iterations', default=-1, type=int, help='number of total iterations to run')
    parser.add_argument('--height', default=512, type=int, help='height of output image')
    parser.add_argument('--width', default=512, type=int, help='width of output image')
    parser.add_argument('-n', '--num_inference_steps', default=50, type=int, help='num inference steps of diffusion model')
    parser.add_argument('-gs', '--guidance_scale', default=7.5, type=float, help='guidance scale of diffusion model')
    parser.add_argument('--latent_path', default=None, type=str, help='path to latent noise file')
    parser.add_argument('--high_precision_steps', default=2, type=int, help='path to latent noise file')
    parser.add_argument('--benchmark', action="store_true", help='benchmark perf of the model')

    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    logging.info(f"Parameters {args}")

    # load model
    if 'xl' in args.model_name_or_path:
        pipe = StableDiffusionXLPipelineMixedPrecision.from_pretrained(
            args.model_name_or_path,
            safety_checker=None,
            add_watermarker=False,
        )
        quantize_ops = ['conv2d']
    else:
        pipe = StableDiffusionPipelineMixedPrecision.from_pretrained(args.model_name_or_path)
        quantize_ops = ['conv2d', 'linear']
    if torch.cuda.is_available():
        pipe.to('cuda:0')

    # data type
    dtype = torch.float32
    if args.precision == "fp32":
        print("Running fp32 ...")
    elif args.precision == "bf16":
        print("Running bf16 ...")
        dtype = torch.bfloat16
    elif args.precision == "int8" or args.precision == "int8-bf16":
        print(f"Running {args.precision} ...")
        if args.precision == "int8-bf16":
            dtype = torch.bfloat16
            unet_bf16 = copy.deepcopy(pipe.unet).to(device=pipe.unet.device, dtype=dtype)
            pipe.unet_bf16 = unet_bf16
            pipe.HIGH_PRECISION_STEPS = args.high_precision_steps
        from quantization_modules import load_int8_model
        pipe.unet = load_int8_model(
            pipe.unet, args.int8_model_path,
            'fake' in args.int8_model_path, quantize_ops=quantize_ops,
            convert=not torch.cuda.is_available()
        )
        if torch.cuda.is_available():
            pipe.unet.to('cuda:0')
    else:
        raise ValueError("--precision needs to be the following:: fp32, bf16, fp16, int8, int8-bf16")
    pipe.to(dtype=dtype)
    if args.benchmark:
        prompt = "An astronaut riding a green horse"
        if args.latent_path:
            generator = None
            latents = torch.load(args.latent_path).to(dtype)
        else:
            generator = torch.manual_seed(args.seed)
            latents = None
        context = torch.cpu.amp.autocast(dtype=dtype) \
            if not torch.cuda.is_available() else torch.cuda.amp.autocast(dtype=dtype)
        def inference():
            if args.precision == "bf16":
                with context, torch.no_grad():
                    output = pipe(
                        prompt, generator=generator, latents=latents,
                        guidance_scale=args.guidance_scale, num_inference_steps=args.num_inference_steps,
                        height=args.height, width=args.width, output_type="np"
                    ).images
            else:
                with torch.no_grad():
                    output = pipe(
                        prompt, generator=generator, latents=latents,
                        guidance_scale=args.guidance_scale, num_inference_steps=args.num_inference_steps,
                        height=args.height, width=args.width, output_type="np"
                    ).images
        print("Warm up.")
        inference()
        test_num = 10
        import time
        all_times = []
        print("Benchmark start.")
        for i in range(test_num):
            start_time = time.time()
            inference()
            end_time = time.time()
            all_times.append(end_time-start_time)
        all_times = sorted(all_times)[1:-1]
        avg_time = round(np.mean(all_times) * 100) / 100
        print(f"Average inference time of {args.precision} is {avg_time}s.")
        exit()

    # prepare dataloader
    val_coco = dset.CocoCaptions(root = '{}/val2017'.format(args.dataset_path),
                                 annFile = '{}/annotations/captions_val2017.json'.format(args.dataset_path),
                                 transform=transforms.Compose([transforms.Resize((512, 512)), transforms.PILToTensor(), ]))

    val_dataloader = torch.utils.data.DataLoader(val_coco,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=0,
                                                 sampler=None)

    print("Running accuracy ...")
    # run model
    fid = FrechetInceptionDistance(normalize=True)
    if torch.cuda.is_available():
        fid.inception.to('cuda:0')
    for i, (images, prompts) in enumerate(val_dataloader):
        prompt = prompts[0][0]
        real_image = images[0]
        print(f"prompt {i+1}: ", prompt)
        if args.latent_path:
            generator = None
            latents = torch.load(args.latent_path).to(dtype)
        else:
            generator = torch.manual_seed(args.seed)
            latents = None
        image_name = f"{i+1}_" + prompt.replace('\n', '').replace(' ', '_').replace('/', '') # time.strftime("%Y%m%d_%H%M%S")
        fake_image_path = f"{args.output_dir}/fake_image_{image_name}.png"
        real_image_path = f"{args.output_dir}/real_image_{image_name}.png"
        if os.path.exists(fake_image_path):
            fake_image = Image.open(fake_image_path)
            output = (np.array(fake_image.getdata()).reshape(fake_image.size[0], fake_image.size[1], 3)/255.0, )
        else:
            if args.precision == "bf16":
                context = torch.cpu.amp.autocast(dtype=dtype) \
                    if not torch.cuda.is_available() else torch.cuda.amp.autocast(dtype=dtype)
                with context, torch.no_grad():
                    output = pipe(
                        prompt, generator=generator, latents=latents,
                        guidance_scale=args.guidance_scale, num_inference_steps=args.num_inference_steps,
                        height=args.height, width=args.width, output_type="np"
                    ).images
            else:
                with torch.no_grad():
                    output = pipe(
                        prompt, generator=generator, latents=latents,
                        guidance_scale=args.guidance_scale, num_inference_steps=args.num_inference_steps,
                        height=args.height, width=args.width, output_type="np"
                    ).images

        if args.output_dir:
            if not os.path.exists(args.output_dir):
                os.mkdir(args.output_dir)
            if not os.path.exists(fake_image_path):
                Image.fromarray((output[0] * 255).round().astype("uint8")).save(fake_image_path)
            if not os.path.exists(real_image_path):
                Image.fromarray(real_image.permute(1, 2, 0).numpy()).save(real_image_path)

        fake_image = torch.tensor(output[0]).unsqueeze(0).permute(0, 3, 1, 2)
        real_image = real_image.unsqueeze(0) / 255.0

        fid.update(real_image, real=True)
        fid.update(fake_image, real=False)

        if args.iterations > 0 and i == args.iterations - 1:
            break
    res = f"FID: {float(fid.compute())}"
    with open(os.path.join(args.output_dir, 'fid.txt'), mode='w') as f:
        f.write(res)
    print(res)

if __name__ == '__main__':
    main()