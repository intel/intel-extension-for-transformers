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

def main(args_in: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="run quantization and inference")
    parser.add_argument(
        "model", type=Path, help="directory containing model file or model id"
    )

    # quantization related arguments.
    parser.add_argument(
        "--bits",
        type=int,
        help="number of bits to use for quantization (default: 4)",
        default=4,
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

    # inference related arguments.
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="prompt to start generation with (default: empty)",
        default="",
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
    path = Path(parent_path, "convert_model.py")
    convert_cmd = "python {} --outtype f32 --outfile {} {}".format(
        path,
        Path(work_path, "ne_{}_f32.bin".format(model_type)),
        args.model
    )
    print("convert model ...")
    os.system(convert_cmd)

    # 2. quantize
    # TODO: if quantize
    path = Path(parent_path, "quant_bin.py")
    cmd = "python {} --model_name {} --model_file {} --out_file {} --bits {} --block_size {} --scale_dtype {} --compute_type {}".format(
        path,
        model_type,
        Path(work_path, "ne_{}_f32.bin".format(model_type)),
        Path(work_path, "ne_{}_{}.bin".format(model_type, args.bits, args.block_size)),
        args.bits,
        args.block_size,
        args.scale_dtype,
        args.compute_type
    )
    print("quantize model ...")
    print(cmd)
    os.system(cmd)

    # 3. inference
    path = Path(parent_path, "chat_llm.py")
    cmd = "python {} --model_name {} -m {} --seed 12 -c 512 -b 1024 -n 256 --keep 48 -t 56 --repeat_penalty 1.0 --color -p \"{}\"".format(
        path,
        model_type,
        Path(work_path, "ne_{}_{}.bin".format(model_type, args.bits, args.block_size)),
        args.prompt,
    )
    print("inferce model ...")
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    main()
