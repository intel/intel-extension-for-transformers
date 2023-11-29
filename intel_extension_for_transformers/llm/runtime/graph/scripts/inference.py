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

model_maps = {"gpt_neox": "gptneox", "llama2": "llama", "gpt_bigcode": "starcoder"}
build_path = Path(Path(__file__).parent.absolute(), "../build/")

def is_win():
    return sys.platform.startswith('win')

def main(args_in: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="main program llm running")
    parser.add_argument("--model_name", type=str, help="Model name: String", required=True)
    parser.add_argument("-m", "--model", type=Path, help="Path to the executed model: String", required=True)
    parser.add_argument(
        "--build_dir", type=Path, help="Path to the build file: String", default=build_path
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="Prompt to start generation with: String (default: empty)",
        default="",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="The path of the chatglm tokenizer: String (default: THUDM/chatglm-6b)",
        default="THUDM/chatglm-6b",
    )
    parser.add_argument(
        "-n",
        "--n_predict",
        type=int,
        help="Number of tokens to predict: Int (default: 0, -1 = all)",
        default=-1,
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        help="Number of threads to use during computation: Int (default: 56)",
        default=56,
    )
    parser.add_argument(
        "-b",
        "--batch_size_truncate",
        type=int,
        help="Batch size for prompt processing: Int (default: 512)",
        default=512,
    )
    parser.add_argument(
        "-c",
        "--ctx_size",
        type=int,
        help="Size of the prompt context: Int (default: 512, can not be larger than specific model's context window length)",
        default=512,
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="NG seed: Int (default: -1, use random seed for < 0)",
        default=-1,
    )
    parser.add_argument(
        "--repeat_penalty",
        type=float,
        help="Penalize repeat sequence of tokens: Float (default: 1.1, 1.0 = disabled)",
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
        help="Number of tokens to keep from the initial prompt: Int (default: 0, -1 = all)",
        default=0,
    )
    parser.add_argument(
        "--shift-roped-k",
        action="store_true",
        help="Use ring-buffer and thus do not re-computing after reaching ctx_size (default: False)",
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
    if is_win():
        path = Path(args.build_dir, "./Bin/Release/run_{}.exe".format(model_name))
    else:
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
    if args.shift_roped_k:
        cmd.extend(["--shift-roped-k"])
    if args.color:
        cmd.append("--color")
    if args.memory_f32:
        cmd.extend(["--memory-f32"])
    if args.memory_f16:
        cmd.extend(["--memory-f16"])
    if args.memory_auto:
        cmd.extend(["--memory-auto"])


    if (args.model_name == "chatglm"):
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        token_ids_list = tokenizer.encode(args.prompt)
        token_ids_list = map(str, token_ids_list)
        token_ids_str = ', '.join(token_ids_list)
        cmd.extend(["--ids", token_ids_str])
    elif (args.model_name == "baichuan" or args.model_name == "qwen"):
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        def truncate(ids, max_length):
            """Truncates a list of integers to a given length.

            Args:
                ids: A list of integers.
                max_length: The maximum length of the truncated list.
            """
            if len(ids) > max_length:
                ids = ids[:max_length]

        def encode_history(history, max_length=4096):
            """Encodes a history of text into a list of integers.

            Args:
                history: A list of strings, representing the history of text.
                max_length: The maximum length of the encoded history.

            Returns:
                A list of integers, representing the encoded history.
            """

            assert len(history) % 2 == 1, "invalid history size {}".format(len(history))

            USER_TOKEN_ID = 195
            ASSISTANT_TOKEN_ID = 196

            ids = []
            for i in range(len(history)):
                if i % 2 == 0:
                    ids.append(USER_TOKEN_ID)
                else:
                    ids.append(ASSISTANT_TOKEN_ID)

                content_ids = tokenizer.encode(args.prompt)
                ids.extend(content_ids)

            ids.append(ASSISTANT_TOKEN_ID)
            truncate(ids, max_length)
            return ids

        history = [args.prompt]
        token_ids_list = encode_history(history)
        token_ids_list = map(str, token_ids_list)
        token_ids_str = ', '.join(token_ids_list)
        cmd.extend(["--ids", token_ids_str])

    print("cmd:", cmd)
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
