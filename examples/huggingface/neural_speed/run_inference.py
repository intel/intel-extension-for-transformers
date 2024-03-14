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
from pathlib import Path
from typing import List, Optional
from transformers import AutoTokenizer,TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM, RtnConfig
def main(args_in: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Convert a PyTorch model to a NE compatible file")
    parser.add_argument("--model_path",type=Path,
                        help="model path for local or from hf", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--prompt",type=str,help="model path for local or from hf", default="Once upon a time, there existed a little girl,")
    parser.add_argument("--not_quant" ,action="store_false", help="Whether to use a model with low bit quantization")
    parser.add_argument("--weight_dtype",type=str,
                        help="output weight type, default: int4, we support int4, int8, nf4 and others ", default="int4")
    parser.add_argument("--compute_dtype", type=str, help="compute type", default="int8")
    parser.add_argument("--group_size", type=int, help="group size", default=128)
    parser.add_argument('--use_gptq', action='store_true')
    parser.add_argument("--n_ctx", type=int, help="n_ctx", default=512)
    parser.add_argument("--max_new_tokens", type=int, help="max_new_tokens", default=300)
    args = parser.parse_args(args_in)
    model_name = args.model_path
    woq_config = RtnConfig(load_in_4bit=True, use_quant=args.not_quant,
                                       weight_dtype=args.weight_dtype, compute_dtype=args.compute_dtype, group_size=args.group_size, use_gptq=args.use_gptq)
    prompt = args.prompt
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    streamer = TextStreamer(tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").input_ids

    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=woq_config, trust_remote_code=True)
 
    outputs = model.generate(inputs, streamer=streamer, ctx_size=args.n_ctx, max_new_tokens=args.max_new_tokens)



if __name__ == "__main__":
    main()
    
