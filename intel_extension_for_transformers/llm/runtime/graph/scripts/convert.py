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
import numpy as np
from pathlib import Path
import argparse
from typing import List, Optional
from transformers import AutoConfig
import subprocess

model_maps = {"gpt_neox": "gptneox", "gpt_bigcode": "starcoder"}


def convert_model(model, outfile, outtype):
    import pdb; pdb.set_trace()
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    model_type = model_maps.get(config.model_type, config.model_type)

    gpt_model = 'gptq' in str(model).lower()
    if gpt_model:
        path = Path(Path(__file__).parent.absolute(), "convert_gptq_{}.py".format(model_type))
    else:
        path = Path(Path(__file__).parent.absolute(), "convert_{}.py".format(model_type))
    cmd = []
    cmd.extend(["python", path])
    cmd.extend(["--outfile", outfile])
    cmd.extend(["--outtype", outtype])
    cmd.extend([model])

    print("cmd:", cmd)
    subprocess.run(cmd)

def main(args_in: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert a PyTorch model to a NE compatible file"
    )
    parser.add_argument(
        "--outtype",
        choices=["f32", "f16"],
        help="output format, default: f32",
        default="f32",
    )
    parser.add_argument("--outfile", type=Path, required=True, help="path to write to")
    parser.add_argument(
        "model", type=Path, help="directory containing model file or model id"
    )
    args = parser.parse_args(args_in)

    if args.model.exists():
        dir_model = args.model.as_posix()
    else:
        dir_model = args.model

    convert_model(dir_model, args.outfile, args.outtype)


if __name__ == "__main__":
    main()
