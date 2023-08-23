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

model_maps = {"gpt_neox": "gptneox"}

def main(args_in: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="main program for chatting with llm")
    parser.add_argument("--model_name", type=str, help="model name", required=True)
    parser.add_argument("-m", "--model", type=Path, help="path ne model", required=True)
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="prompt to start generation with (default: empty)",
        default="",
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
        "--batch_size",
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
        help="number of tokens to keep from the initial prompt (default: 0, -1 = all)  ",
        default=0,
    )

    args = parser.parse_args(args_in)
    print(args)
    model_name = model_maps.get(args.model_name, args.model_name)
    path = Path(
        Path(__file__).parent.absolute(), "../build/bin/chat_{}".format(model_name)
    )
    if not path.exists():
        print("Please build graph first or select the correct model name.")
        sys.exit(1)

    cmd = "{} --model {} --prompt \"{}\" --n-predict {} --threads {} --batch-size {} \
        --ctx-size {} --seed {} --repeat-penalty {} --keep {}".format(
        path,
        args.model,
        args.prompt,
        args.n_predict,
        args.threads,
        args.batch_size,
        args.ctx_size,
        args.seed,
        args.repeat_penalty,
        args.keep,
    )
    if args.color:
        cmd = cmd + " --color"
    print("cmd:", cmd)
    os.system(cmd)


if __name__ == "__main__":
    main()
