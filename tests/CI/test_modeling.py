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

import copy
import os
import shutil
import torch
import unittest
from intel_extension_for_transformers.transformers.modeling.modeling_seq2seq import INCModelForSeq2SeqLM
from neural_compressor import PostTrainingQuantConfig, quantization
from optimum.intel.neural_compressor import INCConfig
from optimum.exporters import TasksManager
from torch.jit import RecursiveScriptModule
from transformers import (
    AutoTokenizer,
)


os.environ["WANDB_DISABLED"] = "true"
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "true"
MODEL_NAME = "t5-small"


def get_seq2seq_example_inputs(model):
    onnx_config_class = TasksManager.get_exporter_config_constructor(model_type=model.config.model_type, exporter="onnx", task="text2text-generation")
    onnx_config = onnx_config_class(model.config, use_past=model.config.use_cache, use_past_in_inputs=model.config.use_cache)
    encoder_onnx_config = onnx_config.with_behavior("encoder")
    decoder_onnx_config = onnx_config.with_behavior("decoder", use_past=False)
    decoder_with_past_onnx_config = onnx_config.with_behavior("decoder", use_past=True, use_past_in_inputs=model.config.use_cache)
    encoder_dummy_inputs = encoder_onnx_config.generate_dummy_inputs(framework="pt")
    decoder_dummy_inputs = decoder_onnx_config.generate_dummy_inputs(framework="pt")
    decoder_dummy_inputs["encoder_outputs"] = tuple(decoder_dummy_inputs["encoder_outputs"][0:1])
    del decoder_dummy_inputs["attention_mask"]
    decoder_with_past_dummy_inputs = decoder_with_past_onnx_config.generate_dummy_inputs(framework="pt")
    decoder_with_past_dummy_inputs["encoder_outputs"] = tuple(decoder_with_past_dummy_inputs["encoder_outputs"][0:1])
    decoder_with_past_dummy_inputs["past_key_values"] = tuple(decoder_with_past_dummy_inputs["past_key_values"])
    del decoder_with_past_dummy_inputs["attention_mask"]
    return encoder_dummy_inputs, decoder_dummy_inputs, decoder_with_past_dummy_inputs

class TestModeling(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.prompt_texts = ["Translate to German: My name is Arthur"]
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=4)
        self.input_ids = self.tokenizer(self.prompt_texts[0], return_tensors="pt").input_ids
        input_bs, input_len = self.input_ids.shape
        self.attention_mask = torch.ones(input_bs, input_len)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree('./mlruns', ignore_errors=True)
        shutil.rmtree('./quantized_model', ignore_errors=True)

    def test_seq2seq_jit_model(self):
        model = INCModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        input_ids = self.tokenizer(self.prompt_texts[0], return_tensors="pt").input_ids
        gen_ids = model.generate(input_ids, max_new_tokens=32, **self.generate_kwargs)
        jit_model = INCModelForSeq2SeqLM.from_pretrained(MODEL_NAME, export=True)
        jit_gen_ids = jit_model.generate(input_ids, max_new_tokens=32, **self.generate_kwargs)
        self.assertTrue(torch.equal(gen_ids, jit_gen_ids))
        self.assertTrue(isinstance(jit_model.encoder_model, RecursiveScriptModule))

    def test_seq2seq_model_quant_with_ipex(self):
        # Will add IPEX backend after IPEX fixed jit model inference issue.
        for backend in ("default",):
            model = INCModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
            encoder_dummy_inputs, decoder_dummy_inputs, decoder_with_past_dummy_inputs = \
                get_seq2seq_example_inputs(model)
            def encoder_calib_func(prepared_model):
                prepared_model(
                    input_ids=self.input_ids,
                    attention_mask=self.attention_mask,
                )

            encoder_conf = PostTrainingQuantConfig(
                backend=backend,
                example_inputs=encoder_dummy_inputs,
            )
            model.encoder_model.config.return_dict = False
            encoder_model = quantization.fit(
                model.encoder_model,
                encoder_conf,
                calib_func=encoder_calib_func,
            )
            model.encoder_model = encoder_model
            if backend == "ipex":
                def decoder_with_past_calib_func(prepared_model):
                    model.decoder_with_past_model = prepared_model
                    model.generate(
                        input_ids=self.input_ids,
                        attention_mask=self.attention_mask,
                        max_new_tokens=32,
                        **self.generate_kwargs,
                    )
                decoder_with_past_conf = PostTrainingQuantConfig(
                    backend=backend,
                    example_inputs=decoder_with_past_dummy_inputs,
                )
                model.decoder_model.config.return_dict = False
                decoder_with_past_model = quantization.fit(
                    copy.deepcopy(model.decoder_model),
                    decoder_with_past_conf,
                    calib_func=decoder_with_past_calib_func,
                )
                model.decoder_with_past_model = decoder_with_past_model
            def decoder_calib_func(prepared_model):
                model.decoder_model = prepared_model
                model.generate(
                    input_ids=self.input_ids,
                    attention_mask=self.attention_mask,
                    max_new_tokens=32,
                    **self.generate_kwargs,
                )
            decoder_conf = PostTrainingQuantConfig(
                backend=backend,
                example_inputs=decoder_dummy_inputs,
            )
            decoder_model = quantization.fit(
                copy.deepcopy(model.decoder_model),
                decoder_conf,
                calib_func=decoder_calib_func,
            )
            model.decoder_model = decoder_model
            if backend == "ipex":
                model.config.torchscript = True
            model.config.torch_dtype = "int8"

            model.save_pretrained('./quantized_model')
            inc_config = INCConfig(quantization=encoder_conf, save_onnx_model=False)
            inc_config.save_pretrained('./quantized_model')

            loaded_model = INCModelForSeq2SeqLM.from_pretrained('./quantized_model')
            if backend == "ipex":
                self.assertTrue(isinstance(loaded_model.encoder_model, RecursiveScriptModule))
            else:
                self.assertTrue(isinstance(loaded_model.encoder_model.embed_tokens,
                                           torch.fx.graph_module.GraphModule))


if __name__ == "__main__":
    unittest.main()
