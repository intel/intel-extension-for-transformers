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
import os
import sys
from pathlib import Path
import argparse
from typing import List, Optional
import subprocess
from transformers import AutoTokenizer

model_maps = {"gpt_neox": "gptneox", "llama2": "llama"}
build_path = Path(Path(__file__).parent.absolute(), "../build/")

def main(args_in: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="main program llm running")
    parser.add_argument("--model_name", type=str, help="model name", required=True)
    parser.add_argument("-m", "--model", type=Path, help="path ne model", required=True)
    parser.add_argument(
        "--build_dir", type=Path, help="path to build directory", default=build_path
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="prompt to start generation with (default: empty)",
        default="",
    )
    parser.add_argument(
        "--glm_tokenizer",
        type=str,
        help="the path of the chatglm tokenizer",
        default="THUDM/chatglm-6b",
    )
    parser.add_argument(
        "-n",
        "--n_predict",
        type=int,
        help="number of tokens to predict (default: -1, -1 = infinity)",
        default=-1,
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        help="number of threads to use during computation (default: 56)",
        default=56,
    )
    parser.add_argument(
        "-b",
        "--batch_size_truncate",
        type=int,
        help="batch size for prompt processing (default: 512)",
        default=512,
    )
    parser.add_argument(
        "-c",
        "--ctx_size",
        type=int,
        help="size of the prompt context (default: 512)",
        default=512,
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="NG seed (default: -1, use random seed for < 0)",
        default=-1,
    )
    parser.add_argument(
        "--repeat_penalty",
        type=float,
        help="penalize repeat sequence of tokens (default: 1.1, 1.0 = disabled)",
        default=1.1,
    )
    parser.add_argument(
        "--color",
        action="store_true",
        help="colorise output to distinguish prompt and user input from generations",
    )
    parser.add_argument(
        "--keep",
        type=int,
        help="number of tokens to keep from the initial prompt (default: 0, -1 = all)",
        default=0,
    )
    parser.add_argument(
        "--memory-f32",
        action="store_true",
        help="Use fp32 for the data type of kv memory",
    )
    parser.add_argument(
        "--memory-f16",
        action="store_true",
        help="Use fp16 for the data type of kv memory",
    )
    parser.add_argument(
        "--memory-auto",
        action="store_true",
        help="Try with jblas flash attn managed format for kv memory (Currently GCC13 & AMX required); "
        "fall back to fp16 if failed (default option for kv-memory)",
    )

    args = parser.parse_args(args_in)
    print(args)
    model_name = model_maps.get(args.model_name, args.model_name)
    path = Path(args.build_dir, "./bin/run_{}".format(model_name))
    if not path.exists():
        print("Please build graph first or select the correct model name.")
        sys.exit(1)

    cmd = [path]
    cmd.extend(["--model",          args.model])
    cmd.extend(["--prompt",         args.prompt])
    cmd.extend(["--n-predict",      str(args.n_predict)])
    cmd.extend(["--threads",        str(args.threads)])
    cmd.extend(["--batch-size-truncate",     str(args.batch_size_truncate)])
    cmd.extend(["--ctx-size",       str(args.ctx_size)])
    cmd.extend(["--seed",           str(args.seed)])
    cmd.extend(["--repeat-penalty", str(args.repeat_penalty)])
    cmd.extend(["--keep",           str(args.keep)])
    if args.color:
        cmd.append(" --color")
    if args.memory_f32:
        cmd.extend(["--memory-f32"])
    if args.memory_f16:
        cmd.extend(["--memory-f16"])
    if args.memory_auto:
        cmd.extend(["--memory-auto"])


    if (args.model_name == "chatglm"):
        tokenizer = AutoTokenizer.from_pretrained(args.glm_tokenizer, trust_remote_code=True)
        token_ids_list = tokenizer.encode(args.prompt)
        token_ids_list = map(str, token_ids_list)
        token_ids_str = ', '.join(token_ids_list)
        print(token_ids_str)
        cmd.extend(["--ids", token_ids_str])

    print("cmd:", cmd)
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
