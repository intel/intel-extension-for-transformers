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
import os
import psutil
import signal

class SadTalker():
    """Faster Talking Face Animation."""
    def __init__(self, device="cpu"):
        # prepare the models
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(cur_dir)
        scripts_dir = os.path.join(cur_dir, "scripts")
        download_script = f"{scripts_dir}/download_models.sh"
        subprocess.run(["bash", download_script])
        self.device = device

    def convert(self, source_image, driven_audio, output_video_path="./response.wav",
                bf16=False, result_dir="./results", p_num=1, enhancer="gfpgan"):
        if self.device == "cpu":
            self.convert_cpu(source_image, driven_audio, output_video_path=output_video_path,
                bf16=bf16, result_dir=result_dir, p_num=p_num, enhancer=enhancer)
        elif self.device == "cuda":
            self.convert_gpu(source_image, driven_audio, output_video_path=output_video_path,
                result_dir=result_dir, enhancer=enhancer)
        else:
            raise Exception("Hardware not supported!")

    def convert_cpu(self, source_image, driven_audio, output_video_path="./response.mp4",
                bf16=False, result_dir="./results", p_num=1, enhancer="gfpgan"):
        multi_instance_cmd = ""
        core_num = psutil.cpu_count(logical=False)
        unit = core_num / p_num
        for i in range(p_num):
            start_core = (int)(i * unit)
            end_core = (int)((i+1) * unit - 1)
            bf16 = "" if not bf16 else "--bf16"
            enhancer_str = "" if not enhancer else f"--enhancer={enhancer}"
            # compose the command for instance parallelism
            multi_instance_cmd += f"numactl -l -C {start_core}-{end_core} python inference.py --driven_audio {driven_audio} --source_image {source_image} --result_dir {result_dir} --output_video_path {output_video_path} --cpu --rank={i} --p_num={p_num} {bf16} {enhancer_str} &\n "
        multi_instance_cmd += "wait < <(jobs -p) \nrm -rf logs"
        print(multi_instance_cmd)
        p = subprocess.Popen(multi_instance_cmd, preexec_fn=os.setsid, shell=True, executable='/bin/bash')  # nosec
        try:
            p.communicate()
        except KeyboardInterrupt:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        (output, err) = p.communicate()
        p_status = p.wait()
        print(output)
        print(err)

    def convert_gpu(self, source_image, driven_audio, output_video_path="./response.mp4",
                result_dir="./results", enhancer="gfpgan"):
        enhancer_str = "" if not enhancer else f"--enhancer={enhancer}"
        instance_cmd = f"python inference.py --driven_audio {driven_audio} --source_image {source_image} --result_dir {result_dir} --output_video_path {output_video_path} {enhancer_str} &\n "
        instance_cmd += "wait < <(jobs -p) \nrm -rf logs"
        print(instance_cmd)
        p = subprocess.Popen(instance_cmd, preexec_fn=os.setsid, shell=True, executable='/bin/bash')  # nosec
        try:
            p.communicate()
        except KeyboardInterrupt:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        (output, err) = p.communicate()
        p_status = p.wait()
        print(output)
        print(err)