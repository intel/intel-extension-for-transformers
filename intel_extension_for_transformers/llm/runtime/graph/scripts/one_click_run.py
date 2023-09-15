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
from transformers import AutoConfig
import subprocess

build_path = Path(Path(__file__).parent.absolute(), "../build/")

def main(args_in: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="run quantization and inference")
    parser.add_argument(
        "model", type=Path, help="directory containing model file or model id"
    )
    parser.add_argument(
        "--build_dir", type=Path, help="path to build directory", default=build_path
    )

    # quantization related arguments.
    parser.add_argument(
        "--weight_dtype",
        choices=["int4", "int8"],
        help="weight data type, default: int4",
        default="int4",
    )
    parser.add_argument(
        "--alg",
        type=str,
        help="qquantization algorithm to use: sym/asym (default: sym)",
        default="sym",
    )
    parser.add_argument(
        "--group_size", type=int, help="group size (default: 32)", default=32
    )
    parser.add_argument(
        "--scale_dtype",
        type=str,
        help="fp32/bf16 type for scales (default: fp32)",
        default="fp32",
    )
    parser.add_argument(
        "--compute_type",
        type=str,
        help="Gemm computation data type: int8/fp32/ggml (default: ggml)",
        default="ggml",
    )

    # inference related arguments.
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="prompt to start generation with (default: empty)",
        default="Once upon a time, there existed a ",
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
        help="number of tokens to keep from the initial prompt (default: 0, -1 = all)  ",
        default=0,
    )


    args = parser.parse_args(args_in)


    if args.model.exists():
        dir_model = args.model.as_posix()
    else:
        dir_model = args.model

    parent_path = Path(__file__).parent.absolute()
    config = AutoConfig.from_pretrained(dir_model)
    model_type = config.model_type
    work_path = Path(model_type + "_files")
    if not work_path.exists():
        Path.mkdir(work_path)
    

    # 1. convert
    path = Path(parent_path, "convert.py")
    convert_cmd = ["python", path]
    convert_cmd.extend(["--outfile", Path(work_path, "ne_{}_f32.bin".format(model_type))])
    convert_cmd.extend(["--outtype", "f32"])
    convert_cmd.append(args.model)
    print("convert model ...")
    subprocess.run(convert_cmd)

    # 2. quantize
    path = Path(parent_path, "quantize.py")
    quant_cmd = ["python", path]
    quant_cmd.extend(["--model_name", model_type])
    quant_cmd.extend(["--model_file", Path(work_path, "ne_{}_f32.bin".format(model_type))])
    quant_cmd.extend(["--out_file", Path(work_path, "ne_{}_{}.bin".format(model_type, args.weight_dtype, args.group_size))])
    quant_cmd.extend(["--weight_dtype", args.weight_dtype])
    quant_cmd.extend(["--group_size", str(args.group_size)])
    quant_cmd.extend(["--scale_dtype", args.scale_dtype])
    quant_cmd.extend(["--compute_type", args.compute_type])
    quant_cmd.extend(["--build_dir", args.build_dir])
    print("quantize model ...")
    subprocess.run(quant_cmd)

    # 3. inference
    path = Path(parent_path, "inference.py")
    infer_cmd = ["python", path]
    infer_cmd.extend(["--model_name", model_type])
    infer_cmd.extend(["-m", Path(work_path, "ne_{}_{}.bin".format(model_type, args.weight_dtype, args.group_size))])
    infer_cmd.extend(["--prompt", args.prompt])
    infer_cmd.extend(["--n_predict",      str(args.n_predict)])
    infer_cmd.extend(["--threads",        str(args.threads)])
    infer_cmd.extend(["--batch-size-truncate",     str(args.batch_size_truncate)])
    infer_cmd.extend(["--ctx_size",       str(args.ctx_size)])
    infer_cmd.extend(["--seed",           str(args.seed)])
    infer_cmd.extend(["--repeat_penalty", str(args.repeat_penalty)])
    infer_cmd.extend(["--keep",           str(args.keep)])
    infer_cmd.extend(["--build_dir", args.build_dir])
    print("inferce model ...")
    subprocess.run(infer_cmd)


if __name__ == "__main__":
    main()
