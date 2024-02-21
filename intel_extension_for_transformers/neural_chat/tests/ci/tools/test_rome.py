#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import transformers
from intel_extension_for_transformers.transformers import MixedPrecisionConfig
from intel_extension_for_transformers.neural_chat import build_chatbot, PipelineConfig
from intel_extension_for_transformers.neural_chat.models.model_utils import MODELS
from intel_extension_for_transformers.neural_chat.tools.rome import ROMEHyperParams, apply_rome_to_model
import unittest

LLAMA2_7B_CHAT_MODEL = "fxmarty/tiny-llama-fast-tokenizer"

class TestROME(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_rome(self):
        seed = 42
        checkpointing = True
        requests = [
            {
                "prompt": "{} is located in the city of",
                "subject": "Eiffel Tower",
                "target": " Rome",
                "queries": [
                "Where is Eiffel Tower? ",
                "The Eiffel Tower is located at "
                ]
            },
        ]
        queries = [query for request in requests for query in request["queries"]]
        batch_first = True
        transformers.set_seed(seed)

        chatbot = build_chatbot(
            PipelineConfig(model_name_or_path=LLAMA2_7B_CHAT_MODEL,
                           optimization_config=MixedPrecisionConfig(dtype="float32"))
        )
        model = MODELS[chatbot.model_name]["model"]
        tokenizer = MODELS[chatbot.model_name]["tokenizer"]
        batch_first = True
        if checkpointing:
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable()
            model.config.use_cache = False

        print("#"*9 + "Get hyperparameters" + "#"*9)
        hparams = ROMEHyperParams.from_name('llama-7b')
        hparams.layers = [0]
        hparams.v_loss_layer = 1
        hparams.mom2_n_samples = 300
        print(hparams)

        pre_update_text = [chatbot.predict(query) for query in queries]

        print("#"*9 + "Applying rome to model" + "#"*9)
        model_new, _ = apply_rome_to_model(
            model,
            tokenizer,
            requests,
            hparams,
            batch_first,
            return_diff_weights=False
        )
        MODELS[chatbot.model_name]["model"] = model_new

        post_update_text = [chatbot.predict(query) for query in queries]
        print("#"*9 + "Generated pre-update text" + "#"*9)
        print("\n\n".join([queries[i] + " " + pre_update_text[i] for i in range(len(queries))]))
        print("#"*9 + "Generated post-update text" + "#"*9)
        print("\n\n".join([queries[i] + " " + post_update_text[i] for i in range(len(queries))]))


if __name__ == "__main__":
    unittest.main()
