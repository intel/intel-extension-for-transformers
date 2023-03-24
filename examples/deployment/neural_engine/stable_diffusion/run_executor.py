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
import os


def benchmark(pipe, neural_engine_graph):
    print('benchmark start...')
    warmup = 3
    total = 7
    total_time = 0
    with torch.no_grad():
        prompt = "a photo of an astronaut riding a horse on mars"
        for i in range(total):
            start2 = time.time()
            pipe(prompt, engine_graph=neural_engine_graph).images[0]
            end2 = time.time()
            if i >= warmup:
                total_time += end2 - start2
            print("Total inference latency: ", str(end2 - start2) + "s")
    print("Average Latency: ", (total_time) / (total - warmup), "s")
    print("Average Throughput: {:.5f} samples/sec".format((total - warmup) / (total_time)))


def accuracy(pipe, original_pipe, neural_engine_graph):
    with torch.no_grad():
        prompt = "a photo of an astronaut riding a horse on mars"

        engine_image = pipe(prompt, engine_graph=neural_engine_graph).images[0]
        engine_image.save("astronaut_rides_horse_from_engine.png")

        pytorch_image = original_pipe(prompt).images[0]
        pytorch_image.save("astronaut_rides_horse_from_pytorch.png")

        engine_image_dir = "engine_image"
        os.makedirs(engine_image_dir, exist_ok=True)
        if os.path.isfile(os.path.join(engine_image_dir, "astronaut_rides_horse.png")):
            os.remove(os.path.join(engine_image_dir, "astronaut_rides_horse.png"))
        engine_image.save(engine_image_dir + "/astronaut_rides_horse.png")

        pytorch_image_dir = "pytorch_image"
        os.makedirs(pytorch_image_dir, exist_ok=True)
        if os.path.isfile(os.path.join(pytorch_image_dir, "astronaut_rides_horse.png")):
            os.remove(os.path.join(pytorch_image_dir, "astronaut_rides_horse.png"))
        pytorch_image.save(pytorch_image_dir + "/astronaut_rides_horse.png")

        fid = fid_score.calculate_fid_given_paths((pytorch_image_dir, engine_image_dir), 1, "cpu", 2048, 2)
        print("Finally FID score Accuracy: {}".format(fid))
        return fid


def executor(pipe, neural_engine_graph, prompt, output_picture_name):
    print('executor start...')
    image = pipe(prompt, engine_graph=neural_engine_graph).images[0]
    image.save(output_picture_name)
    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model",
                        default="CompVis/stable-diffusion-v1-4",
                        type=str,
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--ir_path", default="./ir", type=str, help="Neural engine IR path.")
    parser.add_argument(
        "--prompt",
        default="a photo of an astronaut riding a horse on mars",
        type=str,
        help="The input of the model, like: 'a photo of an astronaut riding a horse on mars'.")
    parser.add_argument("--output_picture_name",
                        default="astronaut_rides_horse.png",
                        type=str,
                        help="output picture name.")
    parser.add_argument("--mode", type=str,
                        help="Benchmark mode of performance or accuracy.")
    return parser.parse_args()


def main():
    args = parse_args()
    pipe = diffusion_utils.StableDiffusionPipeline.from_pretrained(args.input_model)
    neural_engine_graph = diffusion_utils.neural_engine_init(args.ir_path)

    if args.mode == "performance":
        benchmark(pipe, neural_engine_graph)
        return

    if args.mode == "accuracy":
        from diffusers import StableDiffusionPipeline
        original_pipe = StableDiffusionPipeline.from_pretrained(args.input_model)
        accuracy(pipe, original_pipe, neural_engine_graph)
        return

    executor(pipe, neural_engine_graph, args.prompt, args.output_picture_name)

    return


if __name__ == '__main__':
    main()
