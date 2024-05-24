#  Copyright (c) 2024 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import argparse
import time
import os
from vllm import LLM, SamplingParams
from typing import List, Optional
from intel_extension_for_transformers.transformers import AutoModelForCausalLM, RtnConfig
from transformers import AutoTokenizer


def main(args_in: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Model name: String", required=True)
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="Prompt to start generation with: String (default: empty)",
        default="Once upon a time",
    )
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--use_neural_speed", action="store_true")
    args = parser.parse_args(args_in)
    print(args)

    if args.benchmark:
        if args.use_neural_speed:
            os.environ["NEURAL_SPEED_VERBOSE"] = "1"
            woq_config = RtnConfig(bits=4, weight_dtype="int4", compute_dtype="int8", scale_dtype="bf16")
            model_with_ns = AutoModelForCausalLM.from_pretrained(args.model_path, quantization_config=woq_config)

            tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
            inputs = tokenizer(args.prompt, return_tensors="pt").input_ids

            T5 = time.time()
            output = model_with_ns.generate(inputs, max_new_tokens=32)
            T6 = time.time()
            print("neural speed output = ", output)

        llm = LLM(model=args.model_path, trust_remote_code=True)
        sampling_params = SamplingParams(max_tokens=32)
        T1 = time.time()
        original_outputs = llm.generate(args.prompt, sampling_params)  # Generate texts from the prompts.
        T2 = time.time()
        vllm_latency = (T2 - T1) * 1000

        model = AutoModelForCausalLM.from_pretrained(args.model_path, use_vllm=True)
        T3 = time.time()
        optimized_output = model.generate(args.prompt, sampling_params)
        T4 = time.time()
        qbits_latency = (T4 - T3) * 1000

        print("original outputs = ", original_outputs)
        print("input_tokens_length = ", len(original_outputs[0].prompt_token_ids))
        print("output_tokens_length = ", len(original_outputs[0].outputs[0].token_ids))

        print("optimized outputs = ", optimized_output)
        print("input_tokens_length = ", len(optimized_output[0].prompt_token_ids))
        print("output_tokens_length = ", len(optimized_output[0].outputs[0].token_ids))

        print('The qbits optimized generate:%.2f ms' % qbits_latency)
        print('The original vLLM   generate:%.2f ms' % vllm_latency)

        return

    model = AutoModelForCausalLM.from_pretrained(args.model_path, use_vllm=True)
    output = model.generate(args.prompt)
    print(output)


if __name__ == "__main__":
    main()
