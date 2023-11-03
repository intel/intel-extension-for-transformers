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

model_maps = {"gpt_neox": "gptneox", "llama2": "llama", "gpt_bigcode": "starcoder"}
build_path = Path(Path(__file__).parent.absolute(), "../build/")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(args_in: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Quantize weights of NE files")
    parser.add_argument("--model_name", type=str, help="model name", required=True)
    parser.add_argument(
        "--model_file", type=Path, help="Path to the fp32 model: String", required=True
    )
    parser.add_argument(
        "--out_file", type=Path, help="Path to the quantized model: String", required=True
    )
    parser.add_argument(
        "--build_dir", type=Path, help="Path to the build file: String", default=build_path
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to the configuration file: String (default: \"\")",
        default="",
    )
    parser.add_argument(
        "--nthread", type=int, help="Number of threads to use: Int (default: 1)", default=1
    )
    parser.add_argument(
        "--weight_dtype",
        choices=["int4", "int8"],
        help="Data type of quantized weight: int4/int8 (default: int4)",
        default="int4",
    )
    parser.add_argument(
        "--alg",
        type=str,
        help="Quantization algorithm to use: sym/asym (default: sym)",
        default="sym",
    )
    parser.add_argument(
        "--group_size", type=int, help="Group size: Int (default: 32)", default=32
    )
    parser.add_argument(
        "--scale_dtype",
        type=str,
        help="Data type of scales: bf16/fp32 (default: fp32)",
        default="fp32",
    )
    parser.add_argument(
        "--compute_dtype",
        type=str,
        help="Data type of Gemm computation: int8/bf16/fp32 (default: int8)",
        default="int8",
    )
    parser.add_argument(
        "--use_ggml",
        action="store_true",
        help="enable ggml for quantization and inference",
    )
    args = parser.parse_args(args_in)

    model_name = model_maps.get(args.model_name, args.model_name)
    path = Path(args.build_dir, "./bin/quant_{}".format(model_name))
    if not path.exists():
        print(path)
        print("Please build graph first or select the correct model name.")
        sys.exit(1)

    cmd = [path]
    cmd.extend(["--model_file",     args.model_file])
    cmd.extend(["--out_file",       args.out_file])
    cmd.extend(["--nthread",        str(args.nthread)])
    cmd.extend(["--weight_dtype",   str(args.weight_dtype)])
    cmd.extend(["--alg",            args.alg])
    cmd.extend(["--group_size",     str(args.group_size)])
    cmd.extend(["--scale_dtype",    args.scale_dtype])
    cmd.extend(["--compute_dtype",  args.compute_dtype])
    if args.use_ggml:
        cmd.extend(["--use_ggml"])
    
    print(cmd)
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
