#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
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

import os
from transformers import PyTorchBenchmark
from transformers.benchmark.benchmark import *
from transformers.benchmark.benchmark_utils import *
from .model import OptimizedModel


def _prepare_inference_func(self, model_name: str, batch_size: int, sequence_length: int) -> Callable[[], None]:
    config = self.config_dict[model_name]

    if self.args.torchscript:
        config.torchscript = True

    logger.warning("Function transformers.PyTorchBenchmark._prepare_inference_func is replaced "
                    "by intel_extension_for_transformers.optimization.benchmark to support int8 models.")
    model = OptimizedModel.from_pretrained(model_name)

    model.eval()
    model.to(self.args.device)

    # encoder-decoder has vocab size saved differently
    vocab_size = config.vocab_size if hasattr(config, "vocab_size") else config.encoder.vocab_size
    input_ids = torch.randint(vocab_size, (batch_size, sequence_length), dtype=torch.long, device=self.args.device)

    if self.args.fp16:   # pragma: no cover 
        logger.info("Running training in Mixed Precision...")
        if not self.args.is_gpu:
            raise ValueError("Mixed precision is possible only for GPU.")
        # amp seems to have memory leaks so that memory usage
        # is measured using .half() for now https://github.com/NVIDIA/apex/issues/439
        model.half()

    if self.args.torchscript:
        with torch.no_grad():
            try:
                inference_model = torch.jit.trace(model, input_ids)
            except:
                inference_model = torch.jit.trace(model, input_ids, strict=False)
    else:
        inference_model = model

    def encoder_decoder_forward():   # pragma: no cover 
        with torch.no_grad():
            outputs = inference_model(input_ids, decoder_input_ids=input_ids)
        return outputs

    def encoder_forward():
        with torch.no_grad():
            outputs = inference_model(input_ids)
        return outputs

    def get_weight_size(model):
        if isinstance(model, torch.jit.ScriptModule):
            torch.jit.save(model, "temp.p")
        else:
            torch.save(model.state_dict(), "temp.p")
        weight_size = os.path.getsize("temp.p") / 1024 / 1024
        os.remove('temp.p')
        return weight_size

    if not hasattr(self, 'weight_size_dict'):
        self.weight_size_dict = dict()

    if model_name not in self.weight_size_dict:
        weight_size = round(get_weight_size(inference_model), 3)
        self.weight_size_dict.update({model_name: weight_size})

    _forward = encoder_decoder_forward if config.is_encoder_decoder else encoder_forward
    return _forward

PyTorchBenchmark._prepare_inference_func = _prepare_inference_func


origin_func = PyTorchBenchmark.run
def run(self):
    output = origin_func(self)
    self.print_fn("\n" + 20 * "=" + ("INFERENCE - MODEL SIZE - RESULT").center(40) + 20 * "=")
    self.print_fn(80 * "-")
    self.print_fn("Model Name".center(60) + "Model Size in MB".center(15))
    self.print_fn(80 * "-")
    for model_name in self.args.model_names:
        self.print_fn(model_name[:50].center(60), 
                      str(self.weight_size_dict[model_name]).center(15),)
    self.print_fn(80 * "-")
    return output

PyTorchBenchmark.run = run


ExecutorBenchmarkArguments = PyTorchBenchmarkArguments


class ExecutorBenchmark(PyTorchBenchmark):

    args: ExecutorBenchmarkArguments
    configs: PretrainedConfig
    framework: str = "Executor"

    def _prepare_inference_func(self, model_name: str, batch_size: int, sequence_length: int) -> Callable[[], None]:
        config = self.config_dict[model_name]

        logger.warning("Function transformers.PyTorchBenchmark._prepare_inference_func is replaced "
                        "by intel_extension_for_transformers.optimization.benchmark to support executor.")

        from intel_extension_for_transformers.backends.neural_engine.compile import compile
        model = compile(model_name)

        # encoder-decoder has vocab size saved differently
        vocab_size = config.vocab_size if hasattr(config, "vocab_size") else config.encoder.vocab_size

        input_ids = torch.randint(
            vocab_size, 
            (batch_size, sequence_length),
        ).int()
        if len(model.nodes[0].output_tensors) == 2:
            attention_mask = torch.ones((batch_size, sequence_length)).int()
            input_data = [input_ids, attention_mask]
        elif len(model.nodes[0].output_tensors) == 3:   # pragma: no cover 
            attention_mask = torch.ones((batch_size, sequence_length)).int()
            token_type_ids = torch.ones((batch_size, sequence_length)).int()
            input_data = [input_ids, attention_mask, token_type_ids]
        else:   # pragma: no cover 
            input_data = [input_ids]

        def encoder_forward():
            with torch.no_grad():
                outputs = model.inference(input_data)
            return outputs

        def get_weight_size(model_name):
            weight_size = os.path.getsize(model_name) / 1024 / 1024
            return weight_size

        if not hasattr(self, 'weight_size_dict'):
            self.weight_size_dict = dict()

        if model_name not in self.weight_size_dict:
            weight_size = round(get_weight_size(model_name), 3)
            self.weight_size_dict.update({model_name: weight_size})

        _forward = encoder_forward
        return _forward
