# Copyright (c) 2024 Intel Corporation
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

"""
A model worker executes the model.
"""
import argparse
import asyncio
import dataclasses
import logging
import json
import time
from typing import List, Union
import threading
import uuid

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, PlainTextResponse
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, GenerationConfig, StoppingCriteria, StoppingCriteriaList
import torch
import uvicorn

from constants import WORKER_HEART_BEAT_INTERVAL
from inference import load_model, generate_stream
from utils import (build_logger, server_error_msg, pretty_print_semaphore)

from asr import AudioSpeechRecognition
from tts import TextToSpeech

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")
global_counter = 0

model_semaphore = None


def heart_beat_worker(controller):

    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


class ModelWorker:
    def __init__(self, controller_addr, worker_addr,
                 worker_id, no_register, model_path, model_name,
                 device, num_gpus, load_8bit=False, itrex=False, ipex=False):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        self.model_name = model_name or model_path.split("/")[-1]
        self.device = device

        logger.info(f"Loading the model {self.model_name} on worker {worker_id} ...")
        self.model, self.tokenizer = load_model(
            model_path, device, num_gpus, load_8bit, itrex, ipex)

        if hasattr(self.model.config, "max_sequence_length"):
            self.context_len = self.model.config.max_sequence_length
        elif hasattr(self.model.config, "max_position_embeddings"):
            self.context_len = self.model.config.max_position_embeddings
        else:
            self.context_len = 2048

        self.generate_stream_func = generate_stream

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,))
            self.heart_beat_thread.start()

    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status()
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(f"Send heart beat. Models: {[self.model_name]}. "
                    f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
                    f"global_counter: {global_counter}")

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(url, json={
                    "worker_name": self.worker_addr,
                    "queue_length": self.get_queue_length()}, timeout=5)
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + len(
                model_semaphore._waiters)

    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    def generate_stream_gate(self, params):
        try:
            for output in self.generate_stream_func(self.model, self.model_name, self.tokenizer,
                    params, self.device, self.context_len, args.stream_interval):
                ret = {
                    "text": output,
                    "error_code": 0,
                }
                yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.OutOfMemoryError:
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"


app = FastAPI()


def release_model_semaphore():
    model_semaphore.release()


@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    generator = worker.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_model_semaphore)
    return StreamingResponse(generator, background=background_tasks, media_type="text/event-stream")

class StopOnTokens(StoppingCriteria):
    def __init__(self, min_length: int, start_length: int, stop_token_id: list[int]):
        self.min_length = min_length
        self.start_length = start_length
        self.stop_token_id = stop_token_id

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        if scores is not None:
            if len(scores) > self.min_length:
                for stop_id in self.stop_token_id:
                    if input_ids[0][self.start_length - 1 + len(scores)] == stop_id:
                        return True
        elif input_ids.shape[-1] - self.start_length > self.min_length:
            for stop_id in self.stop_token_id:
                if input_ids[0][input_ids.shape[-1] - 1] == stop_id:
                    return True
        return False

@app.post("/talkingbot", response_class=PlainTextResponse)
async def talkingbot(request: Request):
    params = await request.json()
    saved_path = params["file_name"]
    voice = params["voice"]
    # audio -> text
    logger.info("1: audio --> text")
    text = asr.audio2text(saved_path)
    logger.info(text)
    prompt = """Have a conversation with a human. You must generate suitable response in short to the user input.\n### Input:\n{}\n### Response:""".format(text)
    # text -> answer
    logger.info("2: text --> answer")
    worker.tokenizer.pad_token = worker.tokenizer.eos_token
    stop_token_ids = [worker.model.model.generation_config.eos_token_id]
    stop_token_ids.append(worker.tokenizer(".", return_tensors="pt").input_ids)
    input_tokens = worker.tokenizer.batch_encode_plus([prompt], return_tensors="pt", padding=True)
    input_token_len = input_tokens.input_ids.shape[-1]

    stop = StopOnTokens(min_length=44, start_length=input_token_len, stop_token_id=stop_token_ids)
    generation_config = GenerationConfig(
        eos_token_id=0,
        pad_token_id=0,
        use_cache=True,
        min_new_tokens=1,
        max_new_tokens=64,
        temperature=0.9,
        top_p=0.9,
        top_k=1,
        repetition_penalty=1.1,
        num_beams=1,
        early_stopping=True,
        ## use default decode mode
    )
    generation_kwargs = dict(
                    generation_config=generation_config, return_dict_in_generate=True
                )
    generation_kwargs["stopping_criteria"] = StoppingCriteriaList([stop])
    with torch.no_grad():
        with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
            ## worker.model ==> ipexwrapper
            output = worker.model.model.generate(**input_tokens, **generation_kwargs)
    generated_texts = worker.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
    logger.info("raw generated texts", generated_texts)
    if "### Response:" in generated_texts:
        generated_texts = generated_texts.split("### Response:")[1].strip()
    lines = generated_texts.split('\n')
    result_lines = []
    for line in lines:
        if 'Input:' in line or '```python' in line:
            break
        result_lines.append(line)
    generated_texts = '\n'.join(result_lines)
    generated_texts = generated_texts.replace('#', '')
    generated_texts = generated_texts.split('include <')[0]
    # answer -> audio
    # answer -> audio
    logger.info("3: answer --> audio")
    answer_speech_path = tts.text2speech(generated_texts, voice=voice)
    logger.info("Done!!!")
    logger.info(answer_speech_path)
    return answer_speech_path


@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()

def validate_port(value):
    try:
        port = int(value)
        if 1 <= port <= 65535:
            return port
        else:
            raise argparse.ArgumentTypeError("Port number must be between 1 and 65535.")
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid port number. Must be an integer.")

def validate_device(value):
    valid_devices = ["cpu", "cuda", "mps"]
    if value in valid_devices:
        return value
    else:
        raise argparse.ArgumentTypeError(f"Invalid device. Must be one of {', '.join(valid_devices)}.")

def validate_limit_model_concurrency(value):
    if value >= 0:
        return value
    else:
        raise argparse.ArgumentTypeError("Limit model concurrency must be a non-negative integer.")

def validate_stream_interval(value):
    if value > 0:
        return value
    else:
        raise argparse.ArgumentTypeError("Stream interval must be a positive integer.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=validate_port, default=8080)
    parser.add_argument("--worker-address", type=str,
        default="http://localhost:8080")
    parser.add_argument("--controller-address", type=str,
        default="http://localhost:80")
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--device", type=validate_device, choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--limit-model-concurrency", type=validate_limit_model_concurrency, default=5)
    parser.add_argument("--stream-interval", type=validate_stream_interval, default=2)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--ipex", action="store_true")
    parser.add_argument("--itrex", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = ModelWorker(args.controller_address,
                         args.worker_address,
                         worker_id,
                         args.no_register,
                         args.model_path,
                         args.model_name,
                         args.device,
                         args.num_gpus,
                         args.load_8bit,
                         args.itrex,
                         args.ipex)
    asr = AudioSpeechRecognition()
    tts = TextToSpeech()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
