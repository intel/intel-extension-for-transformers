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
import torch
import unittest
import shutil
import torch.utils.data as data
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from intel_extension_for_transformers.transformers.modeling import AutoModelForCausalLM
from intel_extension_for_transformers.llm.quantization.nn.modules import QuantizedLinearQBits, QuantizedLoraLinearQBits
from intel_extension_for_transformers.llm.quantization.utils import convert_to_quantized_model, replace_linear
from intel_extension_for_transformers.llm.utils.generation import _beam_search, _greedy_search
from intel_extension_for_transformers.transformers import WeightOnlyQuantConfig


class DummyDataset(data.Dataset):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(llama_model_path)
        self.sequences = "Where is intel-extension-for-transformers based? NYC or SH? " + \
            "intel-extension-for-transformers is based in SH."
        self.encoded_dict = self.tokenizer(self.sequences)
        self.encoded_dict['labels'] = self.encoded_dict['input_ids']

    def __len__(self):
        return 1

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        return self.encoded_dict

class M(torch.nn.Module):
    def __init__(self, with_bias=False):
        super().__init__()
        self.linear = torch.nn.Linear(32, 2, bias=with_bias)

    def forward(self, x):
        return self.linear(x)


llama_model_path = "fxmarty/tiny-llama-fast-tokenizer"


class TestWeightOnly(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.workspace = "./woq_tmp"
        # if workspace not exist, create it
        if not os.path.exists(cls.workspace):
            os.mkdir(cls.workspace)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.workspace, ignore_errors=True)
        shutil.rmtree('tmp', ignore_errors=True)

    def test_woq_config(self):
        config = WeightOnlyQuantConfig(weight_dtype="int4_fullrange", group_size=32)
        diff_res = config.to_diff_dict()
        ref_config = {'weight_dtype': 'int4_fullrange'}
        self.assertEqual(diff_res, ref_config)
        print(diff_res)
        print(config.to_dict())
        print(config.to_json_string())
        config.to_json_file(f"{self.workspace}/config.json")
        print(config)

    def test_woq_config_post_init_runtime(self):
        config = WeightOnlyQuantConfig(weight_dtype="fp4", compute_dtype="int8", scheme="asym", scale_dtype="fp8")
        config.post_init_runtime()
        config_dict = config.to_dict()
        self.assertEqual(config_dict["weight_dtype"], "fp4_e2m1")
        self.assertEqual(config_dict["compute_dtype"], "fp32")
        self.assertEqual(config_dict["scheme"], "sym")
        self.assertEqual(config_dict["scale_dtype"], "fp32")
        config.to_json_file(f"{self.workspace}/config_post_init_runtime.json")
        print(config)

    def test_int8(self):
        raw_wei = torch.rand(2, 32, dtype=torch.float)
        compress_wei = torch.ops.bestlaop.woq_quantize(raw_wei, True, 32, "fp32", "int8", "fp32", False)
        revert_wei = torch.zeros(2, 32, dtype=torch.float)
        torch.ops.bestlaop.woq_dequantize(compress_wei, revert_wei, True, "fp32", "int8", "fp32")
        for bias in [True, False]:
            model = M(with_bias=bias)
            with torch.no_grad():
                model.linear.weight = torch.nn.Parameter(revert_wei)
            activation = torch.rand(1, 32, dtype=torch.float)
            output = model(activation)

            config = WeightOnlyQuantConfig(weight_dtype="int8", group_size=32)
            config.post_init()
            convert_to_quantized_model(model, config)
            output_quant = model(activation)
            print(output)
            print(output_quant)
            assert torch.allclose(output, output_quant, rtol=0.01)

    def test_int4(self):
        raw_wei = torch.rand(2, 32, dtype=torch.float)
        compress_wei = torch.ops.bestlaop.woq_quantize(raw_wei, True, 32, "fp32", "int4_fullrange", "fp32", False)
        revert_wei = torch.zeros(2, 32, dtype=torch.float)
        torch.ops.bestlaop.woq_dequantize(compress_wei, revert_wei, True, "fp32", "int4_fullrange", "fp32")
        for bias in [True, False]:
            model = M(with_bias=bias)
            with torch.no_grad():
                model.linear.weight = torch.nn.Parameter(revert_wei)
            activation = torch.rand(1, 5, 32, dtype=torch.float)
            output = model(activation)
            with torch.no_grad():
                model.linear.weight = torch.nn.Parameter(raw_wei)

            config = WeightOnlyQuantConfig(weight_dtype="int4_fullrange", group_size=32)
            config.post_init()
            convert_to_quantized_model(model, config)
            output_quant = model(activation)
            print(output)
            print(output_quant)
            assert torch.allclose(output, output_quant, rtol=0.01)

    def test_auto_model(self):
        model = AutoModelForCausalLM.from_pretrained(llama_model_path, load_in_4bit=True, use_neural_speed=False)
        module_list = []
        for name, module in model.named_modules():
            if isinstance(module, QuantizedLinearQBits):
                module_list.append(name)
        self.assertTrue(len(module_list) > 0)
        tokenizer = AutoTokenizer.from_pretrained(llama_model_path)
        prompt = "how to test the code?"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        bound_method_1 = _greedy_search.__get__(model, model.__class__)
        setattr(model, "greedy_search", bound_method_1)
        bound_method_2 = _beam_search.__get__(model, model.__class__)
        setattr(model, "beam_search", bound_method_2)
        model.config.token_latency = True
        output = model.generate(
            input_ids, max_new_tokens=int(5), num_beams=1
        )
        self.assertTrue(len(output) == 2 and isinstance(output[1], list))
        output = model.generate(
            input_ids, max_new_tokens=int(5), num_beams=2
        )
        self.assertTrue(len(output) == 2 and isinstance(output[1], list))

    def test_auto_model_with_config(self):
        config = WeightOnlyQuantConfig()
        model = AutoModelForCausalLM.from_pretrained(llama_model_path,
                                                     quantization_config=config,
                                                     use_neural_speed=False)
        module_list = []
        for name, module in model.named_modules():
            if isinstance(module, QuantizedLinearQBits):
                module_list.append(name)
        self.assertTrue(len(module_list) > 0)

    def test_auto_model_saving_loading(self):
        model = AutoModelForCausalLM.from_pretrained(llama_model_path, load_in_4bit=True, use_neural_speed=False)
        module_list = []
        for name, module in model.named_modules():
            if isinstance(module, QuantizedLinearQBits):
                module_list.append(name)
        self.assertTrue(len(module_list) > 0)
        model.save_pretrained(self.workspace, safe_serialization=False)
        loaded_model = AutoModelForCausalLM.from_pretrained(self.workspace)
        for name, module in loaded_model.named_modules():
            if isinstance(module, QuantizedLinearQBits):
                module_list.append(name)
        self.assertTrue(len(module_list) > 0)

    def test_nf4_training(self):
        model = AutoModelForCausalLM.from_pretrained(llama_model_path, load_in_4bit=True, use_neural_speed=False)
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=None,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        lora_weights = {}
        for name, module in model.named_modules():
            if isinstance(module, QuantizedLoraLinearQBits) and "nf4" in module.weight_dtype:
                lora_weights[name] = [
                    getattr(module.lora_A, module.active_adapter[0]).weight.clone(),
                    getattr(module.lora_B, module.active_adapter[0]).weight.clone()
                ]
        self.assertTrue(len(lora_weights) > 0)

        trainer = Trainer(model=model,
                          train_dataset=DummyDataset(),
                          eval_dataset=DummyDataset(),
                          args=TrainingArguments(output_dir='tmp',
                                                 logging_steps=50,
                                                 num_train_epochs=1000,
                                                 learning_rate=1e-4))
        trainer.train()
        for name, module in model.named_modules():
            if isinstance(module, QuantizedLoraLinearQBits) and "nf4" in module.weight_dtype:
                self.assertTrue((lora_weights[name][0] != getattr(module.lora_A,
                                                                  module.active_adapter[0]).weight).any())
                self.assertTrue((lora_weights[name][1] != getattr(module.lora_B,
                                                                  module.active_adapter[0]).weight).any())
                module.merge()
                module.unmerge()
        model.merge_and_unload()

    def test_int8_training(self):
        model = AutoModelForCausalLM.from_pretrained(llama_model_path, load_in_8bit=True, use_neural_speed=False)
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=None,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        lora_weights = {}
        for name, module in model.named_modules():
            if isinstance(module, QuantizedLoraLinearQBits) and "int8" in module.weight_dtype:
                lora_weights[name] = [
                    getattr(module.lora_A, module.active_adapter[0]).weight.clone(),
                    getattr(module.lora_B, module.active_adapter[0]).weight.clone()
                ]
        self.assertTrue(len(lora_weights) > 0)

        trainer = Trainer(model=model,
                          train_dataset=DummyDataset(),
                          eval_dataset=DummyDataset(),
                          args=TrainingArguments(output_dir='tmp',
                                                 logging_steps=50,
                                                 num_train_epochs=1000,
                                                 learning_rate=1e-4))
        trainer.train()
        for name, module in model.named_modules():
            if isinstance(module, QuantizedLoraLinearQBits) and "int8" in module.weight_dtype:
                self.assertTrue((lora_weights[name][0] != getattr(module.lora_A,
                                                                  module.active_adapter[0]).weight).any())
                self.assertTrue((lora_weights[name][1] != getattr(module.lora_B,
                                                                  module.active_adapter[0]).weight).any())

if __name__ == "__main__":
    unittest.main()
