# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import fire
import json
import transformers
from typing import Optional
from intel_extension_for_transformers.transformers import MixedPrecisionConfig
from intel_extension_for_transformers.neural_chat import build_chatbot, PipelineConfig
from intel_extension_for_transformers.neural_chat.models.model_utils import MODELS
from intel_extension_for_transformers.neural_chat.tools.rome import ROMEHyperParams, apply_rome_to_model

def print_head(x, pad=3):
    r"""
    Prints a string with # box for emphasis.

    Example:

    ############################
    #  Applying ROME to model  #
    ############################
    """

    n = len(x)
    print()
    print("".join(["#" for _ in range(n + 2 * pad)]))
    print(
        "#"
        + "".join([" " for _ in range(pad - 1)])
        + x
        + "".join([" " for _ in range(pad - 1)])
        + "#"
    )
    print("".join(["#" for _ in range(n + 2 * pad)]))

def test_rome(
    data: str, model: str, config: str, checkpointing: Optional[bool] = False, seed: Optional[int] = 99
) -> None:
    r"""
    Edits a pre-trained model using model-editing algorithms.

    Args:
        data (`str`):
            The path of the `json` file containing the samples for editing.
        model (`str`):
            The name or path of the pre-trained transformer model to be edited.
        config (`str`):
            The name of the hyper-parameters to use for editing the model.
        checkpointing (`bool`, *optional*, defaults to `False`):
            Whether to enable gradient checkpointing or not.
        seed (`int`, *optional*, defaults to 42):
            The seed for the random process of the program.
    """

    assert os.path.exists(data), "data not found"

    with open(data, "r", encoding="utf-8") as f:
        requests = json.load(f)

    queries = [query for request in requests for query in request["queries"]]
    batch_first = True
    transformers.set_seed(seed)

    chatbot = build_chatbot(
        PipelineConfig(model_name_or_path=model,
                       optimization_config=MixedPrecisionConfig(dtype="float32"))
    )
    model = MODELS[chatbot.model_name]["model"]
    tokenizer = MODELS[chatbot.model_name]["tokenizer"]
    batch_first = True
    if checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    print_head("Get hyperparameters")
    hparams = ROMEHyperParams.from_name(config)
    print(hparams)

    if len(queries) > 0:
        pre_update_text = [chatbot.predict(query) for query in queries]

    print_head(f"Applying rome to model")
    model_new, _ = apply_rome_to_model(
        model,
        tokenizer,
        requests,
        hparams,
        batch_first,
        return_diff_weights=False
    )
    MODELS[chatbot.model_name]["model"] = model_new

    if len(queries) > 0:
        post_update_text = [chatbot.predict(query) for query in queries]
        print_head("Generated pre-update text")
        print("\n\n".join(["User: " + queries[i] + "\nAssistant: " + pre_update_text[i] for i in range(len(queries))]))
        print_head("Generated post-update text")
        print("\n\n".join(["User: " + queries[i] + "\nAssistant: " + post_update_text[i] for i in range(len(queries))]))


if __name__ == "__main__":
    fire.Fire(test_rome)
