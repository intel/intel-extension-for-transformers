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

import diffusion_utils
import torch
import time
from pytorch_fid import fid_score
from diffusers import DPMSolverMultistepScheduler
import os


def benchmark(pipe, neural_engine_graph, generator):
    print('Benchmark start...')
    warmup = 4
    total = 8
    total_time = 0
    with torch.no_grad():
        prompt = "a photo of an astronaut riding a horse on mars"
        for i in range(total):
            start2 = time.time()
            pipe(prompt, engine_graph=neural_engine_graph, num_inference_steps=20, generator=generator).images[0]
            end2 = time.time()
            if i >= warmup:
                total_time += end2 - start2
            print("Total inference latency: ", str(end2 - start2) + "s")
    print("Average Latency: ", (total_time) / (total - warmup), "s")
    print("Average Throughput: {:.5f} samples/sec".format((total - warmup) / (total_time)))


def accuracy(pipe, original_pipe, neural_engine_graph, generator):
    with torch.no_grad():
        prompt = "a photo of an astronaut riding a horse on mars"

        save_time = time.strftime("_%H_%M_%S")
        # Engine
        engine_image = pipe(prompt, engine_graph=neural_engine_graph, generator=generator).images[0]
        engine_image.save("astronaut_rides_horse_from_engine" + save_time + '.png')

        engine_image_dir = "engine_image"
        os.makedirs(engine_image_dir, exist_ok=True)
        if os.path.isfile(os.path.join(engine_image_dir, "astronaut_rides_horse.png")):
            os.remove(os.path.join(engine_image_dir, "astronaut_rides_horse.png"))
        engine_image.save(engine_image_dir + "/astronaut_rides_horse.png")

        # Pytorch
        pytorch_image = original_pipe(prompt, generator=generator).images[0]
        pytorch_image.save("astronaut_rides_horse_from_pytorch" + save_time + '.png')

        pytorch_image_dir = "pytorch_image"
        os.makedirs(pytorch_image_dir, exist_ok=True)
        if os.path.isfile(os.path.join(pytorch_image_dir, "astronaut_rides_horse.png")):
            os.remove(os.path.join(pytorch_image_dir, "astronaut_rides_horse.png"))
        pytorch_image.save(pytorch_image_dir + "/astronaut_rides_horse.png")

        fid = fid_score.calculate_fid_given_paths((pytorch_image_dir, engine_image_dir), 1, "cpu", 2048, 2)
        print("Finally FID score Accuracy: {}".format(fid))
        return fid


def executor(pipe, neural_engine_graph, prompt, name, size, generator):
    print('Executor start...')
    for i in range(size):
        save_time = time.strftime("_%H_%M_%S")
        image = pipe(prompt, engine_graph=neural_engine_graph, generator=generator).images[0]
        image.save(name + str(i) + save_time + '.png')
    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model",
                        default="runwayml/stable-diffusion-v1-5",
                        type=str,
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument(
        "--prompt",
        default="a photo of an astronaut riding a horse on mars",
        type=str,
        help="The input of the model, like: 'a photo of an astronaut riding a horse on mars'.")
    parser.add_argument("--ir_path", default="./ir", type=str, help="Neural engine IR path.")
    parser.add_argument("--name", default="output_image", type=str, help="output image name.")
    parser.add_argument("--mode", type=str, help="Benchmark mode of latency or accuracy.")
    parser.add_argument("--pipeline", default="text2img", type=str, help="text2img or img2img pipeline.")
    parser.add_argument("--seed", type=int, default=666, help="random seed")
    parser.add_argument("--size", type=int, default=1, help="the number of output images per prompt")
    return parser.parse_args()


def main():
    args = parse_args()
    neural_engine_graph = diffusion_utils.neural_engine_init(args.ir_path)
    if args.pipeline == "text2img":
        dpm = DPMSolverMultistepScheduler.from_pretrained(args.input_model, subfolder="scheduler")
        pipe = diffusion_utils.StableDiffusionPipeline.from_pretrained(args.input_model, scheduler=dpm)
        generator = torch.Generator("cpu").manual_seed(args.seed)
        if args.mode == "latency":
            benchmark(pipe, neural_engine_graph, generator)
            return

        if args.mode == "accuracy":
            from diffusers import StableDiffusionPipeline
            original_pipe = StableDiffusionPipeline.from_pretrained(args.input_model)
            accuracy(pipe, original_pipe, neural_engine_graph, generator)
            return

        executor(pipe, neural_engine_graph, args.prompt, args.name, args.size, generator)

    if args.pipeline == "img2img":
        from diffusion_utils_img2img import StableDiffusionImg2ImgPipeline
        import requests
        from PIL import Image
        from io import BytesIO
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(args.input_model)
        url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
        response = requests.get(url)
        init_image = Image.open(BytesIO(response.content)).convert("RGB")
        init_image = init_image.resize((768, 512))

        prompt = "A fantasy landscape, trending on artstation"
        images = pipe(prompt=prompt, image=init_image, engine_graph=neural_engine_graph, strength=0.75, guidance_scale=7.5).images
        images[0].save("fantasy_landscape.png")

    return


if __name__ == '__main__':
    main()
