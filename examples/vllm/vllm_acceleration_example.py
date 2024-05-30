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

prompt_32 = "Once upone a time, Once upone a time, Once upone a time, Once upone a time, Once upone a time, time",

prompts = [prompt_32]


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
        sampling_params = SamplingParams(max_tokens=32)
        config = RtnConfig(compute_dtype="int8",
                           group_size=128,
                           scale_dtype="bf16",
                           weight_dtype="int4_clip",
                           bits=4)
        print(config)
        llm = LLM(model=args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, use_vllm=True, config=config)

        for prompt in prompts:
            vllm_outputs = llm.generate(prompt, sampling_params)  # Generate texts from the prompts.
            qbits_output = model.generate(prompt, sampling_params)

            print("vLLM input_tokens_length = ", len(vllm_outputs[0].prompt_token_ids),
                  "output_tokens_length = ", len(vllm_outputs[0].outputs[0].token_ids))
            print('The vLLM generate = ',
                  vllm_outputs[0].metrics.finished_time - vllm_outputs[0].metrics.arrival_time, "s")
            print("The vLLM first token time = ",
                  vllm_outputs[0].metrics.first_token_time - vllm_outputs[0].metrics.first_scheduled_time)

            print("QBits_vLLM input_tokens_length = ", len(qbits_output[0].prompt_token_ids),
                  "output_tokens_length = ", len(qbits_output[0].outputs[0].token_ids))
            print('The QBits optimized generate = ',
                  qbits_output[0].metrics.finished_time - qbits_output[0].metrics.arrival_time, "s")
            print("The QBits first token time = ",
                  qbits_output[0].metrics.first_token_time - qbits_output[0].metrics.first_scheduled_time)

            if args.use_neural_speed:
                os.environ["NEURAL_SPEED_VERBOSE"] = "1"
                woq_config = RtnConfig(bits=4, weight_dtype="int4", compute_dtype="int8", scale_dtype="bf16")
                model_with_ns = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                                     quantization_config=woq_config)

                tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
                inputs = tokenizer(args.prompt, return_tensors="pt").input_ids

                output = model_with_ns.generate(inputs, max_new_tokens=32)
                print("neural speed output = ", output)

        return

    model = AutoModelForCausalLM.from_pretrained(args.model_path, use_vllm=True)
    output = model.generate(args.prompt)
    print(output)


if __name__ == "__main__":
    main()
