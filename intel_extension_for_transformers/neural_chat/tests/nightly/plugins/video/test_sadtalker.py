# !/usr/bin/env python
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

from intel_extension_for_transformers.neural_chat.pipeline.plugins.video.face_animation.sadtalker import SadTalker
import unittest
import os
import shutil
import requests
import torch
import subprocess

class TestSadTalker(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        p = subprocess.Popen(["bash", "download_models.sh"])
        p.wait()
        self.cur_directory = os.path.dirname(os.path.abspath(__file__))
        sample_audio_url = "https://github.com/intel/intel-extension-for-transformers/raw/main/intel_extension_for_transformers/neural_chat/assets/audio/welcome.wav"
        sample_img_url = "https://raw.githubusercontent.com/OpenTalker/SadTalker/main/examples/source_image/full_body_2.png"
        img_data = requests.get(sample_img_url).content
        self.source_image = os.path.join(self.cur_directory, "sample_img.jpg")
        self.driven_audio = os.path.join(self.cur_directory, "sample_audio.wav")
        with open(self.source_image, 'wb') as f:
            f.write(img_data)
        audio_data = requests.get(sample_audio_url).content
        with open(self.driven_audio, 'wb') as f:
            f.write(audio_data)
        self.output_video_path = os.path.join(self.cur_directory, "response.mp4")
        self.checkpoint_dir = os.path.join(self.cur_directory, "checkpoints")
        self.enhancer_dir = os.path.join(self.cur_directory, "gfpgan")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sadtalker = SadTalker(device=self.device, checkpoint_dir=self.checkpoint_dir, bf16=True, p_num=4, enhancer=None, output_video_path=self.output_video_path)

    @classmethod
    def tearDownClass(self):
        os.remove(self.output_video_path)
        os.remove(self.source_image)
        os.remove(self.driven_audio)
        shutil.rmtree(self.checkpoint_dir, ignore_errors=True)
        shutil.rmtree(self.enhancer_dir, ignore_errors=True)

    def test_sadtalker_without_enhancer(self):
        self.sadtalker.convert(source_image=self.source_image, driven_audio=self.driven_audio)
        self.assertTrue(os.path.exists(self.output_video_path))

    def test_sadtalker_with_enhancer(self):
        self.sadtalker.enhancer = 'gfpgan'
        self.sadtalker.convert(source_image=self.source_image, driven_audio=self.driven_audio)
        self.assertTrue(os.path.exists(self.output_video_path))

if __name__ == "__main__":
    unittest.main()
