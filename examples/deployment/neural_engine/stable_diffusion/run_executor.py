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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model",
                        default="CompVis/stable-diffusion-v1-4",
                        type=str,
                        help="Input model path.")
    parser.add_argument("--ir_path", default="./ir", type=str, help="onnx model path.")
    parser.add_argument("--prompt",
                        default="a photo of an astronaut riding a horse on mars",
                        type=str,
                        help="the default prompt.")
    parser.add_argument("--name", default="astronaut_rides_horse.png", type=str, help="output picture name.")
    args = parser.parse_args()

    pipe = diffusion_utils.StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4')
    neural_engine_graph = diffusion_utils.neural_engine_init(args.ir_path)

    image = pipe(args.prompt, engine_graph=neural_engine_graph).images[0]

    image.save(args.name)
