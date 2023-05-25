# -*- coding: utf-8 -*
#!/usr/bin/env python3
from gevent import monkey
monkey.patch_all()
import gevent
from bottle import route
from bottle import run
from bottle import request, response, hook

import math
import numpy as np
import string

import time
import json
import threading
import traceback

import io
import base64
import os
import sys
from PIL import Image

import requests
from threading import Condition
import queue
import logging
import intel_extension_for_pytorch as ipex
from config import SERVER_HOST, LOG_DIR, MODEL_PATH, SERVER_PORT
from sql_conn import mysql

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
log_dir = LOG_DIR
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

handler = logging.FileHandler('{0}/server.log'.format(log_dir))
formatter = logging.Formatter('%(levelname)-8s %(asctime)s %(process)-5s %(filename)s[line:%(lineno)d] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

import torch
import subprocess
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline
import time

# model_id = "dicoo_model"
model_id = MODEL_PATH

dpm = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=dpm, torch_dtype=torch.float)

pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, scheduler=dpm, torch_dtype=torch.float)

print("---- enable IPEX ----")
sample = torch.randn(2,4,64,64)
timestep = torch.rand(1)*999
encoder_hidden_status = torch.randn(2,77,768)
input_example = (sample, timestep, encoder_hidden_status)
pipe.unet = ipex.optimize(pipe.unet.eval(), dtype=torch.bfloat16, inplace=True, sample_input=input_example)
pipe_img2img.unet = ipex.optimize(pipe_img2img.unet.eval(), dtype=torch.bfloat16, inplace=True, sample_input=input_example)

class Task():
    """ task class """
    def __init__(self):
        self.cond = Condition()
        self.is_complete = False

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

from io import BytesIO

tq = TaskQueue()
cond = threading.Condition()

@hook('before_request')
def validate():
    REQUEST_METHOD = request.environ.get('REQUEST_METHOD')

    HTTP_ACCESS_CONTROL_REQUEST_METHOD = request.environ.get('HTTP_ACCESS_CONTROL_REQUEST_METHOD')
    if REQUEST_METHOD == 'OPTIONS' and HTTP_ACCESS_CONTROL_REQUEST_METHOD:
        request.environ['REQUEST_METHOD'] = HTTP_ACCESS_CONTROL_REQUEST_METHOD

@hook('after_request')
def enable_cors():
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'PUT, GET, POST, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Authorization, Origin, Accept, Content-Type, X-Requested-With'


@route('/', method=['POST'])
def do_inference():
    #allowed_origins = ["*"]
    #origin = request.headers.get('Origin')
    #if origin in allowed_origins:
    #    response.set_header('Access-Control-Allow-Origin', origin)

    start_time = time.time()

    client_ip = request.environ.get('REMOTE_ADDR')
    print(client_ip)

    try:
        req = json.loads(request.body.read())
    except:
        logger.error("failed to load json of request: {0}".format(request.body.read()))
        return json.dumps(dict(ret_msg="load json failed: {0}".format(request.body.read()), ret_code=4001))

    if not req or not "prompt" in req:
        logger.error("input data format error: {0}".format(request.body.read()))
        return json.dumps(dict(ret_msg="input data format error", ret_code=4002))

    # logger.info("request: {0}".format(req))

    # if req.get("token") is None or req.get("token") != "intel_sd_bf16_112233":
    #     return json.dumps({"msg": "token error: 10011"})

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

    # logger.info("prompt: {}, num_inference_steps: {}, guidance_scale: {}, seed: {}".format(prompt,
    #     num_inference_steps, guidance_scale, seed))

    task = Task()
    task.set_prompt(prompt)
    task.set_steps(num_inference_steps)
    task.set_scale(guidance_scale)
    task.set_seed(seed)
    task.source_img = source_img
    task.strength = strength

    global tq
    tq.push(task)

    """
    try:
        # insert to sql
        result = mysql.insert(req["user"],
                req["timestamp"],
                prompt,
                num_inference_steps,
                guidance_scale, seed,
                "0",
                req["task_type"],
                SERVER_HOST, SERVER_PORT)

        logger.info("insert result: {}".format(result))
    except:
        logger.info("insert error")
        logger.info("prompt: {}, num_inference_steps: {}, guidance_scale: {}, seed: {}".format(prompt,
        num_inference_steps, guidance_scale, seed))
    """

    global cond
    logger.info("producer get lock..............")
    cond.acquire()
    logger.info("producer wake worker...................")
    cond.notify()
    logger.info("producer release lock............")
    cond.release()

    task.wait_for_done()
    end_time = time.time()
    cost_time = end_time - start_time
    logger.info("inference costs: {}".format(cost_time))

    buffered = BytesIO()
    task.response.save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue())
    buffered.close()


    """

    # update status to sql
    try:
        result = mysql.update("1", cost_time, req["user"], req["timestamp"])
        logger.info("update result: {}".format(result))
    except:
        logger.info("update error")
        logger.info("prompt: {}, num_inference_steps: {}, guidance_scale: {}, seed: {}".format(prompt,
        num_inference_steps, guidance_scale, seed))
    """


    return {"img_str": img_b64.decode(), "ip": SERVER_HOST}


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

                if task.source_img is None:
                    with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                        image = pipe(task.prompt,
                                num_inference_steps=task.num_inference_steps,
                                guidance_scale=task.guidance_scale, generator=generator).images[0]
                else:
                    with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                        img_byte = base64.b64decode(task.source_img)
                        init_image = Image.open(BytesIO(img_byte)).convert("RGB")
                        init_image = init_image.resize((768, 512))
                        image = pipe_img2img(prompt=task.prompt, image=init_image,
                                num_inference_steps=task.num_inference_steps,
                                strength=task.strength, guidance_scale=task.guidance_scale, generator=generator).images[0]
                task.is_complete = True
                task.response = image
                task.done()

            except:
                logger.error("exception caught: {}".format(traceback.format_exc()))


if __name__ == "__main__":
    worker = Worker(queue = tq,
                    cond = cond)
    print("[INFO] create main worker done")
    worker.start()
    run(host='0.0.0.0', port=int(SERVER_PORT), debug=True, server='gevent')
