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

model_maps = {"gpt_neox": "gptneox", "llama2": "llama"}


def main(args_in: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Quantize weights of NE files")
    parser.add_argument("--model_name", type=str, help="model name", required=True)
    parser.add_argument(
        "--model_file", type=Path, help="path to the fp32 model", required=True
    )
    parser.add_argument(
        "--out_file", type=Path, help="path to the quantized model", required=True
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="path to the configuration file (default: )",
        default="",
    )
    parser.add_argument(
        "--nthread", type=int, help="number of threads to use (default: 1)", default=1
    )
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
        "--block_size", type=int, help="block size (default: 32)", default=32
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
    args = parser.parse_args(args_in)

    model_name = model_maps.get(args.model_name, args.model_name)
    path = Path(
        Path(__file__).parent.absolute(),
        "../build/bin/quant_{}".format(model_name),
    )
    if not path.exists():
        print(path)
        print("Please build graph first or select the correct model name.")
        sys.exit(1)

    quant_bits = 4
    if args.weight_dtype == "int8":
        quant_bits = 8

    cmd = [path]
    cmd.extend(["--model_file",     args.model_file])
    cmd.extend(["--out_file",       args.out_file])
    cmd.extend(["--nthread",        str(args.nthread)])
    cmd.extend(["--bits",           str(quant_bits)])
    cmd.extend(["--alg",            args.alg])
    cmd.extend(["--block_size",     str(args.block_size)])
    cmd.extend(["--scale_dtype",    args.scale_dtype])
    cmd.extend(["--compute_type",   args.compute_type])
    
    print(cmd)
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
