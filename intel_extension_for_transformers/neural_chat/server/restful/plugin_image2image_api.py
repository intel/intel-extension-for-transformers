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

import base64
from io import BytesIO
import time
import json
import socket
import traceback
from PIL import Image
import torch
from fastapi import APIRouter, Request, BackgroundTasks
from ...cli.log import logger
from ...plugins import plugins

class Image2ImageAPIRouter(APIRouter):

    def __init__(self) -> None:
        super().__init__()
        self.chatbot = None

router = Image2ImageAPIRouter()

class ImageProcessor:
    def __init__(self):
        self.image = None

    def execute_task(self, prompt, num_inference_steps, guidance_scale, seed, source_img):
        try:
            generator = torch.Generator('cpu').manual_seed(seed)
            img_byte = base64.b64decode(source_img)
            init_image = Image.open(BytesIO(img_byte)).convert("RGB")
            init_image = init_image.resize((512, 512))
            self.image = plugins["image2image"]["instance"].image2image(prompt=prompt, image=init_image,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator)
        except Exception as e:
            logger.error("exception caught: {}".format(traceback.format_exc()))


@router.post("/plugin/image2image")
async def do_inference(request: Request):
    start_time = time.time()
    image_processor = ImageProcessor()
    try:
        req = await request.json()
    except Exception as e:
        logger.error(f"failed to load json of request: {0}".format(request))
        return json.dumps({"ret_msg": 'load json failed: {0}'.format(request), "ret_code": 4001})

    if not req or not "prompt" in req:
        logger.error(f"input data format error: {0}".format(request))
        return json.dumps({"ret_msg": 'input data format error', "ret_code": 4002})

    prompt = req["prompt"]
    num_inference_steps = req["steps"]
    guidance_scale = req["guidance_scale"]
    seed = req["seed"]
    if "source_img" in req:
        source_img = req["source_img"]
        strength = req["strength"]
        logger.info("image to image")
    else:
        source_img = None
        strength = 0.0

    background_tasks = BackgroundTasks()
    background_tasks.add_task(image_processor.execute_task, prompt, num_inference_steps,
                              guidance_scale, seed, source_img)

    end_time = time.time()
    cost_time = end_time - start_time
    logger.info("inference costs: {}".format(cost_time))

    buffered = BytesIO()
    image_processor.image.save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue())
    buffered.close()

    return {"img_str": img_b64.decode(), "ip": socket.gethostname()}
