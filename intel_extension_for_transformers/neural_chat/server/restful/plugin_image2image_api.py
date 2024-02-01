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


from fastapi import APIRouter, Request
from io import BytesIO
import time
import json
import socket
import threading
import traceback
import base64
from PIL import Image
from threading import Condition
import queue
import torch
from ...cli.log import logger
from ...plugins import plugins

class Task():
    """ task class """
    def __init__(self):
        self.cond = Condition()
        self.is_complete = False
        self.response = None

    def done(self):
        """ function """
        logger.info("task is done")
        self.cond.acquire()
        self.cond.notify()
        self.cond.release()

    def wait_for_done(self):
        """ function """
        self.cond.acquire()
        self.cond.wait_for(predicate=self.get_task_status, timeout=None)
        self.cond.release()

    def set_prompt(self, prompt):
        """ function """
        self.prompt = prompt

    def set_steps(self, num_inference_steps):
        """ function """
        self.num_inference_steps = num_inference_steps

    def set_scale(self, guidance_scale):
        self.guidance_scale = guidance_scale

    def set_seed(self, seed):
        self.seed = seed

    def set_source_image(self, source_img):
        self.source_img = source_img

    def set_strength(self, strength):
        self.strength = strength

    def set_task_type(self, task_type):
        """ set task type """
        self.task_type = task_type

    def set_start_time(self, time):
        """ set_start_time for time-out """
        self.start_time = time

    def get_task_status(self):
        return self.is_complete

class TaskQueue():
    """ TaskQueue """
    def __init__(self):
        self.queue = queue.Queue()

    def push(self, task):
        """ function """
        self.queue.put(task)

    def pop(self):
        """ function """
        item = self.queue.get_nowait()
        self.queue.task_done()
        return item

    def batch_pop(self, batch_size):
        """ function """
        result = []
        count = 0
        while count < batch_size and not self.queue.empty():
            result.append(self.pop())
            count += 1

        return result

    def empty(self):
        """ function """
        return self.queue.empty()

    def join(self):
        """ function """
        self.queue.join()

class Worker(threading.Thread):
    def __init__(self, queue, cond, batch_size=1):
        threading.Thread.__init__(self)
        self.queue = queue
        self.cond = cond
        self.batch_size = batch_size

    def get_batch_task(self):
        tasks = []
        while len(tasks) < self.batch_size:
            left_size = self.batch_size - len(tasks)
            tmp_tasks = self.queue.batch_pop(left_size)
            if not tmp_tasks:
                break

            tasks.extend(tmp_tasks)
        return tasks

    def run(self):
        # main content of this function
        while True:
            try:
                logger.info("worker get lock..............")
                self.cond.acquire()
                while self.queue.empty():
                    logger.info("worker wait..............")
                    self.cond.wait()
                logger.info("worker release lock..............")
                self.cond.release()

                tasks = self.get_batch_task()
                logger.info("tasks size: {0}".format(len(tasks)))
                if not tasks:
                    continue

                task = tasks[0]
                generator = torch.Generator('cpu').manual_seed(task.seed)
                img_byte = base64.b64decode(task.source_img)
                init_image = Image.open(BytesIO(img_byte)).convert("RGB")
                init_image = init_image.resize((512, 512))
                image = plugins["image2image"]["instance"].image2image(prompt=task.prompt, image=init_image,
                            num_inference_steps=task.num_inference_steps,
                            guidance_scale=task.guidance_scale,
                            generator=generator)
                task.is_complete = True
                task.response = image
                task.done()
            except:
                logger.error("exception caught: {}".format(traceback.format_exc()))


class Image2ImageAPIRouter(APIRouter):

    def __init__(self) -> None:
        super().__init__()
        self.chatbot = None
        self.tq = TaskQueue()
        self.cond = threading.Condition()
        self.worker = Worker(queue = self.tq, cond = self.cond)

router = Image2ImageAPIRouter()


@router.post("/v1/image2image")
async def do_inference(request: Request):
    start_time = time.time()
    try:
        req = await request.json()
    except:
        logger.error("failed to load json of request: {0}".format(request))
        return json.dumps(dict(ret_msg="load json failed: {0}".format(request), ret_code=4001))

    if not req or not "prompt" in req:
        logger.error("input data format error: {0}".format(request))
        return json.dumps(dict(ret_msg="input data format error", ret_code=4002))

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

    task = Task()
    task.set_prompt(prompt)
    task.set_steps(num_inference_steps)
    task.set_scale(guidance_scale)
    task.set_seed(seed)
    task.set_source_image(source_img)
    task.set_strength(strength)

    router.tq.push(task)

    logger.info("producer get lock..............")
    router.cond.acquire()
    logger.info("producer wake worker...................")
    router.cond.notify()
    logger.info("producer release lock............")
    router.cond.release()

    task.wait_for_done()
    end_time = time.time()
    cost_time = end_time - start_time
    logger.info("inference costs: {}".format(cost_time))

    buffered = BytesIO()
    task.response.save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue())
    buffered.close()

    return {"img_str": img_b64.decode(), "ip": socket.gethostname()}
