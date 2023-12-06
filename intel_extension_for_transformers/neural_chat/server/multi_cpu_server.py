# !/usr/bin/env python
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
import os
import json, types
from intel_extension_for_transformers.neural_chat.chatbot import build_chatbot
from intel_extension_for_transformers.neural_chat.config import (
    PipelineConfig, GenerationConfig, LoadingModelConfig
)

from intel_extension_for_transformers.transformers import MixedPrecisionConfig
from intel_extension_for_transformers.neural_chat.server.restful.openai_protocol import ChatCompletionRequest
from intel_extension_for_transformers.neural_chat.server.restful.openai_protocol import ChatCompletionResponse
from intel_extension_for_transformers.neural_chat.server.restful.textchat_api import check_completion_request
from intel_extension_for_transformers.neural_chat.cli.log import logger

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
app = FastAPI(title="NeuralChat SPR Serving Process", description="Serving", version="0.0.1")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-bm", "--base_model_path", type=str, default="")
    parser.add_argument("-pm", "--peft_model_path", type=str, default="")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host ip address",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=80,
        help="Server port number",
    )
    # Add arguments for temperature, top_p, top_k and repetition_penalty
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="The value used to control the randomness of sampling.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.75,
        help="The cumulative probability of tokens to keep for sampling.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="The number of highest probability tokens to keep for sampling.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="The maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--hf_access_token",
        type=str,
        default=None,
        help="Huggingface token to access model",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=0,
        help="The number of beams for beam search.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.1,
        help="The penalty applied to repeated tokens.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.",
    )
    parser.add_argument(
        "--tokenizer_name", type=str, default=None, help="specify tokenizer name"
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="enable when use custom model architecture that is not yet part of \
            the Hugging Face transformers package like MPT",
    )

    # habana parameters
    parser.add_argument(
        "--habana",
        action="store_true",
        help="Whether run on habana",
    )
    parser.add_argument(
        "--use_hpu_graphs",
        action="store_true",
        help="Whether to use HPU graphs or not. Using HPU graphs should give better latencies.",
    )
    parser.add_argument(
        "--use_kv_cache",
        action="store_true",
        help="Whether to use the key/value cache for decoding. It should speed up generation.",
    )
    parser.add_argument(
        "--jit",
        action="store_true",
        help="Whether to use jit trace. It should speed up generation.",
    )
    parser.add_argument(
        "--ipex_int8",
        action="store_true",
        help="Whether to use int8 IPEX quantized model. It should speed up generation.",
    )
    parser.add_argument(
        "--seed",
        default=27,
        type=int,
        help="Seed to use for random generation. Useful to reproduce your runs with `--do_sample`.",
    )
    parser.add_argument(
        "--bad_words",
        default=None,
        type=str,
        nargs="+",
        help="Optional argument list of words that are not allowed to be generated.",
    )
    parser.add_argument(
        "--force_words",
        default=None,
        type=str,
        nargs="+",
        help="Optional argument list of words that must be generated.",
    )
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument(
        "--local_rank", type=int, default=-1, metavar="N", help="Local process rank."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="",
        choices=["completion", "chat", "summarization"],
        help="task name, different task means different templates.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "bfloat16", "float16"],
        default="bfloat16",
        help="bfloat16, float32 or float16",
    )
    parser.add_argument(
        "--return_stats", action='store_true', default=False,)
    args = parser.parse_args()
    return args

args = parse_args()
base_model_path = args.base_model_path

# Check the validity of the arguments
if not 0 < args.temperature <= 1.0:
    raise ValueError("Temperature must be between 0 and 1.")
if not 0 <= args.top_p <= 1.0:
    raise ValueError("Top-p must be between 0 and 1.")
if not 0 <= args.top_k <= 200:
    raise ValueError("Top-k must be between 0 and 200.")
if not 1.0 <= args.repetition_penalty <= 2.0:
    raise ValueError("Repetition penalty must be between 1 and 2.")
if not 0 <= args.num_beams <= 8:
    raise ValueError("Number of beams must be between 0 and 8.")
if not 32 <= args.max_new_tokens <= 1024:
    raise ValueError(
        "The maximum number of new tokens must be between 32 and 1024."
    )

from transformers import set_seed
set_seed(args.seed)

config = PipelineConfig(
    model_name_or_path=base_model_path,
    tokenizer_name_or_path=args.tokenizer_name,
    hf_access_token=args.hf_access_token,
    device="cpu",
    loading_config=LoadingModelConfig(
        use_hpu_graphs=args.use_hpu_graphs,
        cpu_jit=args.jit,
        ipex_int8=args.ipex_int8,
        use_cache=args.use_kv_cache,
        peft_path=args.peft_model_path,
        use_deepspeed=False,
    ),
    optimization_config=MixedPrecisionConfig(dtype=args.dtype)
)
chatbot = build_chatbot(config)
gen_config = GenerationConfig(
    device="cpu",
    task=args.task,
    temperature=args.temperature,
    top_p=args.top_p,
    top_k=args.top_k,
    repetition_penalty=args.repetition_penalty,
    num_beams=args.num_beams,
    max_new_tokens=args.max_new_tokens,
    do_sample=args.temperature > 0.0,
    use_hpu_graphs=args.use_hpu_graphs,
    use_cache=args.use_kv_cache,
    num_return_sequences=args.num_return_sequences,
    ipex_int8=args.ipex_int8
)


# warmup, the first time inference take longer because of graph compilation
for new_text in chatbot.predict_stream(query="Tell me about Intel Xeon.", config=gen_config)[0]:
    if args.local_rank in [-1, 0]:
        print(new_text, end="", flush=True)
print("\n"*3)

@app.post("/v1/chat_completions")
async def chat_completion_endpoint(request: ChatCompletionRequest):
    ret = check_completion_request(request)
    if ret is not None:
        raise RuntimeError("Invalid parameter.")
    try:
        logger.info(f"Predicting chat completion using prompt '{request.prompt}'")
        # Set attributes of the config object from the request
        for attr, value in request.__dict__.items():
            if attr == "stream":
                continue
            setattr(gen_config, attr, value)
        buffered_texts = ""
        if request.stream:
            generator, _ = chatbot.predict_stream(query=request.prompt, config=gen_config)
            if not isinstance(generator, types.GeneratorType):
                generator = (generator,)
            def stream_generator():
                nonlocal buffered_texts
                for output in generator:
                    if isinstance(output, str):
                        chunks = output.split()
                        for chunk in chunks:
                            ret = {
                                "text": chunk,
                                "error_code": 0,
                            }
                            buffered_texts += chunk + ' '
                            yield json.dumps(ret).encode() + b"\0"
                    else:
                        ret = {
                            "text": output,
                            "error_code": 0,
                        }
                        buffered_texts += output + ' '
                        yield json.dumps(ret).encode() + b"\0"
                yield f"data: [DONE]\n\n"
            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            response = chatbot.predict(query=request.prompt, config=gen_config)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    else:
        logger.info(f"Chat completion finished.")
        return ChatCompletionResponse(response=response)

if __name__ == "__main__":
    process_port = args.port + args.local_rank + 1
    try:
        uvicorn.run(app, host=args.host, port=process_port)
    except Exception as e:
        print(f"Error starting uvicorn: {str(e)}")
