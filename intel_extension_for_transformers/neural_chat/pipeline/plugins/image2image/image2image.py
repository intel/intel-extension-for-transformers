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

from .instructpix2pix_pipeline import StableDiffusionInstructPix2PixPipeline
import torch
from .diffusion_utils import neural_engine_init

class Image2Image:
    def __init__(self, bf16_ir_path, device="cpu"):
        self.device = device
        self.pipe_img2img = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                    "timbrooks/instruct-pix2pix", torch_dtype=torch.float32, use_auth_token=True)
        self.neural_engine_graph = neural_engine_init(bf16_ir_path)

    def image2image(self, prompt, image, num_inference_steps, guidance_scale, generator):
        # pylint: disable=E1102
        return self.pipe_img2img(prompt=prompt,
                                 image=image,
                                 engine_graph=self.neural_engine_graph,
                                 num_inference_steps=num_inference_steps,
                                 guidance_scale=guidance_scale,
                                 generator=generator).images[0]
