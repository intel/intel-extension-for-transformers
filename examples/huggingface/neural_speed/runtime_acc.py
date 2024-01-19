#  Copyright (c) 2023 Intel Corporation
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
import sys
import argparse
from intel_extension_for_transformers.llm.evaluation.lm_eval import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate diff for a model")
    parser.add_argument('--model_name', type=str, default="~/Llama-2-7b-chat-hf", help="path to model")
    parser.add_argument('--tasks', type=str, default="lambada_openai")
    parser.add_argument('--model_format', type=str, default="runtime")
    parser.add_argument('--use_gptq', action='store_true')
    args = parser.parse_args()
    print(args)

    results = evaluate(
        model="hf-causal",
        model_args=f'pretrained="{args.model_name}",dtype=float32,use_gptq={args.use_gptq}',
        tasks=[f"{args.tasks}"],
        # limit=5,
        model_format=f"{args.model_format}"
    )

    print(results)
