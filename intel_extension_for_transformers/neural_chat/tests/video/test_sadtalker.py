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

class TestSadTalker(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.cur_directory = os.path.dirname(os.path.abspath(__file__))
        self.assets_path = os.path.join(self.cur_directory, "../../assets")
        sample_img_url = "https://raw.githubusercontent.com/OpenTalker/SadTalker/main/examples/source_image/full_body_2.png"
        data = requests.get(sample_img_url).content
        with open('sample_img.jpg', 'wb') as f:
            f.write(data)
        self.source_image = os.path.join(self.cur_directory, "sample_img.jpg")
        self.driven_audio = os.path.join(self.assets_path, "audio/welcome.wav")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sadtalker = SadTalker(device=self.device)

    @classmethod
    def tearDownClass(self):
        for dir in ['logs', 'enhancer_logs', 'workspace', 'results']:
            shutil.rmtree(dir, ignore_errors=True)
        os.remove('response.mp4')
        os.remove('response2.mp4')
        os.chdir(self.cur_directory)
        os.remove(self.source_image)

    def test_sadtalker_without_enhancer(self):
        self.sadtalker.convert(source_image=self.source_image, driven_audio=self.driven_audio, output_video_path="./response.mp4",
                bf16=True, result_dir="./results", p_num=4, enhancer=None)
        self.assertTrue(os.path.exists("./response.mp4"))

    def test_sadtalker_with_enhancer(self):
        self.sadtalker.convert(source_image=self.source_image, driven_audio=self.driven_audio, output_video_path="./response2.mp4",
                bf16=True, result_dir="./results", p_num=4, enhancer='gfpgan')
        self.assertTrue(os.path.exists("./response2.mp4"))

if __name__ == "__main__":
    unittest.main()
