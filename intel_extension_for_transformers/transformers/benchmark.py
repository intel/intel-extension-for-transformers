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

"""Benchmark: provide the inference functions for PyTorchBenchmark and ExecutorBenchmark."""
import os
import sys
import torch
import psutil
from collections import UserDict
from .utils.utility import remove_label
from neural_compressor import __version__ as nc_version
from neural_compressor.utils import logger
from neural_compressor.config import BenchmarkConfig as INCBenchmarkConfig
from packaging import version
from intel_extension_for_transformers.optimization.model import OptimizedModel

if version.parse(nc_version).release < version.parse("2.2").release:
    from neural_compressor.benchmark import _Benchmark as INCBenchmark  # pylint: disable=E0611
else:
    from neural_compressor.benchmark import benchmark_with_raw_cmd  # pylint: disable=E0611

def refactor_batch_size(value, batch_size, old_batch_size=-1):
    """return batched data from value.

    Args:
        value (torch.Tensor): input data.
        batch_size (int): target batch size.
        old_batch_size (int, optional): original batch size of value. Defaults to -1.

    Returns:
        batched_value: batched data.
    """
    batched_value = value
    if isinstance(value, torch.Tensor):
        if old_batch_size == -1:
            old_batch_size = value.shape[0]
        if old_batch_size == value.shape[0]:
            if batch_size <= old_batch_size:
                batched_value = value[:batch_size]
            else:
                tmp_value = value[0].unsqueeze(0)
                for i in range(old_batch_size, batch_size):
                    batched_value = torch.cat((batched_value, tmp_value))
    return batched_value


def get_example_inputs(dataloader, batch_size=1):
    """return batched data from dataloader.

    Args:
        dataloader: dataloader.
        batch_size (int, optional): batch size. Defaults to 1.

    Returns:
        batched_data: batched data.
    """
    it = iter(dataloader)
    data = next(it)
    try:   # pragma: no cover
        for d, label in data:
            data = d
        logger.info("Label is detected in dataloader. If the detection is wrong," + \
                    "please add a fake label to avoid it.")
    except:
        pass
    old_batch_size = dataloader.batch_size
    if isinstance(data, dict) or isinstance(data, UserDict):
        batched_data = {}
        for k, v in data.items():
            batched_data[k] = refactor_batch_size(v, batch_size, old_batch_size)
    elif isinstance(data, list) or isinstance(data, tuple):   # pragma: no cover
        batched_data = []
        for v in data:
            batched_data.append(refactor_batch_size(v, batch_size, old_batch_size))
    else:   # pragma: no cover
        batched_data = refactor_batch_size(data, batch_size, old_batch_size)
    return batched_data


def preprocess_model(model, example_inputs, config, additional_cmd):
    """convert model to torchscript and generate mode.

    Args:
        model (torch.nn.Module): original model.
        example_inputs: used to jit trace original model.
        config (BenchmarkConfig): control the benchmark process.
        additional_cmd (str): additional_cmd for raw_cmd.

    Returns:
        model (torch.jit.ScriptModule): original model.
        example_inputs: preprocessed example_inputs for jit model.
        additional_cmd (str): additional_cmd for raw_cmd.
    """
    if config.generate and not config.torchscript:
        additional_cmd += " --generate"
    if config.torchscript:
        # preprocess input type to tuple or tuple of tensor
        if isinstance(example_inputs, dict) or isinstance(example_inputs, UserDict):
            example_inputs = remove_label(example_inputs)
            example_inputs = tuple(example_inputs.values())
        elif isinstance(example_inputs, list) or isinstance(example_inputs, tuple):
            example_inputs = tuple(example_inputs)

        if config.generate:
            class NewModel(torch.nn.Module):
                def __init__(self, model) -> None:
                    super().__init__()
                    self.model = model
                def forward(self, *args, **kwargs):
                    output = self.model.generate(*args, **kwargs)
                    return output
            model = NewModel(model)
            if isinstance(example_inputs, tuple):
                example_inputs = example_inputs[0] #only input_ids is used.

        with torch.no_grad():
            try:
                model = torch.jit.trace(model, example_inputs)
                model = torch.jit.freeze(model.eval())
            except:   # pragma: no cover
                model = torch.jit.trace(model, example_inputs, strict=False)
                model = torch.jit.freeze(model.eval())
    return model, example_inputs, additional_cmd


def benchmark(model_name_or_path, config=None, example_inputs=None, dataloader=None):
    """function for benchmarking model.

    Args:
        model_name_or_path (str, torch.nn.Module or torch.jit.ScriptModule): model_name_or_path.
        config (BenchmarkConfig, optional): control the benchmark process. Defaults to None.
        example_inputs (optional): used as a input for benchmarking. Defaults to None.
        dataloader (optional): used to build example_inputs for benchmarking. Defaults to None.
    """
    additional_cmd = ""
    from_pretrain_flag = False
    already_jit_flag = False

    # check model type
    if isinstance(model_name_or_path, torch.nn.Module):
        model = model_name_or_path
        if isinstance(model, torch.jit.ScriptModule):
            already_jit_flag = True
    else:
        if config.backend == 'ipex' or config.torchscript == True:
            model = OptimizedModel.from_pretrained(model_name_or_path, torchscript=True)
        else:
            model = OptimizedModel.from_pretrained(model_name_or_path)
        from_pretrain_flag = True
        if isinstance(model, torch.jit.ScriptModule):
            already_jit_flag = True
            from_pretrain_flag = False # jit model can be saved directly.

    # set kwargs to model configuration
    model.eval()
    if hasattr(model, 'config') and config.kwargs is not None:
        for k, v in config.kwargs.items():
            setattr(model.config, k, v)

    # get example inputs
    if example_inputs is None:
        assert dataloader is not None, "Please pass in an example inputs or a dataloder"
        example_inputs = get_example_inputs(dataloader, config.batch_size)
    else:
        example_inputs = refactor_batch_size(example_inputs, config.batch_size)

    # handle generate and torchscript for eager model. not for ipex
    if config.backend == 'ipex':
        additional_cmd += " --enable_ipex"
        # for ipex backend, we will convert model to torchscript if not.
        if not already_jit_flag:
            config.torchscript=True
            import intel_extension_for_pytorch as ipex
            model = ipex.optimize(model)
    if not already_jit_flag:
        model, example_inputs, additional_cmd = preprocess_model(
            model, example_inputs, config, additional_cmd
        )
        if config.torchscript:
            already_jit_flag = True

    # save preprocessed model and preprocessed input.
    tmp_model_path = os.path.join(os.getcwd() + '/tmp_model.bin')
    tmp_data_path = os.path.join(os.getcwd() + '/example_inputs.bin')
    torch.save(example_inputs, tmp_data_path)
    if already_jit_flag:
        model.save(tmp_model_path)
        additional_cmd += " --torchscript"
        # jit model can be saved directly.
        from_pretrain_flag = False
    else:
        try:
            torch.save(model, tmp_model_path)
        except Exception as e:
            assert from_pretrain_flag == True, "Please pass in " + \
              "model_name_or_path instead of torch.nn.module, due to {}".format(e)
            torch.save(model.state_dict(), tmp_model_path)
    weight_size = os.path.getsize(tmp_model_path) / 1024 / 1024
    logger.info("Model size: {} MB".format(weight_size))

    # prepare raw_command for benchmark
    if from_pretrain_flag:
        model_path = model_name_or_path
        additional_cmd += " --from_pretrain"
    else:
        model_path = tmp_model_path
    current_path = os.path.abspath(__file__).rstrip("benchmark.py")
    file_path = current_path + "utils/get_throughput.py \
        --model {} --data {} --batch_size {} --warmup {} \
        --iters {} {}".format(
        model_path, tmp_data_path, config.batch_size, config.warmup,
        config.iteration, additional_cmd
    )
    raw_cmd = "{} {}".format(sys.executable, file_path)

    # trigger INC benchmark
    if config.num_of_instance == -1:
        cpu_counts = psutil.cpu_count(logical=False)
        config.num_of_instance = int(cpu_counts // config.cores_per_instance)
    os.environ['NC_ENV_CONF'] = "False" # mark the start of benchmark
    inc_conf = INCBenchmarkConfig(
        cores_per_instance=config.cores_per_instance,
        num_of_instance=config.num_of_instance
    )
    if version.parse(nc_version).release < version.parse("2.2").release:
        inc_bench = INCBenchmark(inc_conf)
        inc_bench(raw_cmd=raw_cmd)
    else:
        benchmark_with_raw_cmd(raw_cmd, inc_conf)

    # remove tmp files
    os.remove(tmp_model_path)
    os.remove(tmp_data_path)
