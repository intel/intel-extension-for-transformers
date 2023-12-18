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

import argparse
from peft import PeftModel
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description="Load CausalLM and Peft model, then merge and save.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="The model checkpoint for weights initialization."
        "Set to model id of huggingface model hub or local path to the model.",
    )
    parser.add_argument(
        "--peft_name_or_path",
        type=str,
        required=True,
        help="The peft model checkpoint for weights initialization."
        "Set to model id of huggingface model hub or local path to the model.",
    )
    parser.add_argument("--save_path", type=str, default=None, help="Path to save merged model checkpoint.")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, config=config)
    model = PeftModel.from_pretrained(model, args.peft_name_or_path)
    model = model.merge_and_unload()
    save_path = args.save_path
    if save_path is None:
        save_path = "./{}_{}".format(
            args.model_name_or_path.strip('/').split('/')[-1],
            args.peft_name_or_path.strip('/').split('/')[-1])
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print(f"Merged model saved in {save_path}")


if __name__ == "__main__":
    main()
