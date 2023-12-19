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

from intel_extension_for_transformers.neural_chat.pipeline.plugins.audio.utils.video_to_wav import convert_video_to_wav
import os
import argparse
import shlex
import shutil
import unittest
from intel_extension_for_transformers.neural_chat.utils.common import get_device_type

class TestVideo2Wav(unittest.TestCase):
    def setUp(self):
        video_path = \
            "/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/assets/video/"
        if os.path.exists(video_path):
            shutil.rmtree(video_path+"../raw", ignore_errors=True)
            os.mkdir(video_path+"../raw")
        else:
            shutil.rmtree("../assets/raw", ignore_errors=True)
            os.mkdir("../assets/raw")

    def tearDown(self) -> None:
        video_path = \
            "/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/assets/video/"
        if os.path.exists(video_path):
            shutil.rmtree(video_path+"../raw", ignore_errors=True)
        else:
            shutil.rmtree("../assets/raw", ignore_errors=True)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    def test_video_to_wav_file(self):
        parser = argparse.ArgumentParser(__doc__)
        video_path = \
            "/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/assets/video/intel.mp4"
        if os.path.exists(video_path):
            parser.add_argument("--path", type=str, default=video_path)
        else:
            parser.add_argument("--path", type=str, default='../assets/video/intel.mp4')
        parser.add_argument("--is_mono", type=str, default='True')
        parser.add_argument("--sr", type=str, default='16000')
        parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
        args = parser.parse_args()
        output_sample_rate = shlex.quote(args.sr)
        is_exist = os.path.exists(shlex.quote(args.path))
        if not is_exist:
            print("path not existed!")
        else:
            path = shlex.quote(args.path)
            is_mono = shlex.quote(args.is_mono)
            convert_video_to_wav(path, output_sample_rate, is_mono)

        if os.path.exists(video_path):
            video_path_prefix = \
               "/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/assets/video/"
            self.assertTrue(os.path.exists(video_path_prefix+ '../raw/intel.wav'))
        else:
            self.assertTrue(os.path.exists('../assets/raw/intel.wav'))


if __name__ == "__main__":
    unittest.main()
