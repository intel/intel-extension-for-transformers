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
from typing import List, Optional
from intel_extension_for_transformers.transformers import AutoModelForCausalLM


def main(args_in: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Convert a PyTorch model to a NE compatible file")
    parser.add_argument("--model_path", type=str, help="Model name: String", required=True)
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="Prompt to start generation with: String (default: empty)",
        default="Once upon a time",
    )
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args(args_in)
    print(args)

    from intel_extension_for_transformers.transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model_path, use_vllm=True)

    if args.benchmark:
        import time
        from vllm import LLM
        llm = LLM(model=args.model_path, trust_remote_code=True)
        T1 = time.time()
        original_outputs = llm.generate(args.prompt)  # Generate texts from the prompts.
        print("original outputs = ", original_outputs)
        T2 = time.time()

        T3 = time.time()
        optimized_output = model.generate(args.prompt)
        print("optimized outputs = ", optimized_output)
        T4 = time.time()
        print("input_tokens_length) = ", len(optimized_output[0].prompt_token_ids))
        print("output_tokens_length) = ", len(optimized_output[0].outputs[0].token_ids))
        qbits_latency = (T4 - T3) * 1000
        vllm_latency =  (T2 - T1) * 1000
        print('The qbits optimized latency:%.2f ms' % qbits_latency)
        print('The original vLLM   latency:%.2f ms' % vllm_latency)
        print('Latency improves           :%.2f ms' % (vllm_latency - qbits_latency))


        return

    output = model.generate(args.prompt)
    print(output)


if __name__ == "__main__":
    main()
