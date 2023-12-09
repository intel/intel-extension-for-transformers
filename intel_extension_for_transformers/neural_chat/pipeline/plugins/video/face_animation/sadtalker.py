#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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

import subprocess
import sys
import os
import psutil
import signal
import logging
logging.basicConfig(
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    datefmt="%d-%M-%Y %H:%M:%S",
    level=logging.INFO
)
class SadTalker:
    """Faster Talking Face Animation."""

    def __init__(
        self,
        device="cpu",
        checkpoint_dir="./checkpoints",
        bf16=False,
        p_num=1,
        enhancer="gfpgan",
        output_video_path="./response.mp4",
        result_dir="./results",
    ):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.inference_script = os.path.join(cur_dir, "inference.py")
        self.device = device
        self.bf16 = bf16
        self.p_num = p_num
        self.enhancer = enhancer
        self.output_video_path = output_video_path
        self.result_dir = result_dir
        self.checkpoint_dir = checkpoint_dir

    def convert(self, source_image, driven_audio):
        if self.device == "cpu":
            if sys.platform == "linux":
                self.convert_cpu(source_image, driven_audio)
            else:
                raise Exception("Currently only support Linux platform!")
        elif self.device == "cuda":
            self.convert_gpu(source_image, driven_audio)
        else:
            raise Exception("Hardware not supported!")
        return self.output_video_path

    def convert_cpu(self, source_image, driven_audio):
        multi_instance_cmd = ""
        core_num = psutil.cpu_count(logical=False)
        unit = core_num / self.p_num
        for i in range(self.p_num):
            start_core = (int)(i * unit)
            end_core = (int)((i + 1) * unit - 1)
            bf16 = "" if not self.bf16 else "--bf16"
            enhancer_str = "" if not self.enhancer else f"--enhancer {self.enhancer}"
            # compose the command for instance parallelism
            multi_instance_cmd += (
                f"numactl -l -C {start_core}-{end_core} python {self.inference_script} --driven_audio"
                f" {driven_audio} --source_image {source_image} --result_dir {self.result_dir} --output_video_path"
                f" {self.output_video_path} --still --cpu --rank {i} --p_num {self.p_num} {bf16} {enhancer_str}"
                f" --checkpoint_dir {self.checkpoint_dir} &\n "
            )
        multi_instance_cmd += "wait < <(jobs -p) \nrm -rf logs"
        logging.info(multi_instance_cmd)
        p = subprocess.Popen(multi_instance_cmd, preexec_fn=os.setsid, shell=True, executable="/bin/bash")  # nosec
        try:
            p.communicate()
        except KeyboardInterrupt:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        (output, err) = p.communicate()
        p_status = p.wait()
        logging.info(p_status)
        logging.info(output)
        logging.info(err)

    def convert_gpu(self, source_image, driven_audio):
        enhancer_str = "" if not self.enhancer else f"--enhancer {self.enhancer}"
        instance_cmd = (
            f"python {self.inference_script} --driven_audio {driven_audio} --source_image {source_image} --result_dir"
            f" {self.result_dir} --output_video_path {self.output_video_path} {enhancer_str}"
            f" --checkpoint_dir {self.checkpoint_dir} &\n "
        )
        instance_cmd += "wait < <(jobs -p) \nrm -rf logs"
        logging.info(instance_cmd)
        p = subprocess.Popen(instance_cmd, preexec_fn=os.setsid, shell=True, executable="/bin/bash")  # nosec
        try:
            p.communicate()
        except KeyboardInterrupt:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        (output, err) = p.communicate()
        p_status = p.wait()
        logging.info(p_status)
        logging.info(output)
        logging.info(err)
