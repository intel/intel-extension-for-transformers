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

from intel_extension_for_transformers.neural_chat.pipeline.plugins.audio.utils.cut_video import cut_video
import os
import argparse
import shutil
import unittest
from intel_extension_for_transformers.neural_chat.utils.common import get_device_type

class TestCutVideo(unittest.TestCase):
    def setUp(self):
        shutil.rmtree('./raw', ignore_errors=True)
        os.mkdir('./raw')

    def tearDown(self) -> None:
        shutil.rmtree('./raw', ignore_errors=True)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    def test_cut_video(self):
            parser = argparse.ArgumentParser(__doc__)
            video_path = \
              "/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/assets/video/intel.mp4"
            if os.path.exists(video_path):
                parser.add_argument("--path", type=str, default=video_path)
            else:
                parser.add_argument("--path", type=str, default="../assets/video/intel.mp4")
            parser.add_argument("--min", type=str, default='1')
            parser.add_argument("--sr", type=str, default='16000')
            parser.add_argument("--out_path", type=str, default="./raw")
            parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
            args = parser.parse_args()

            # Validate and normalize input and output paths
            if not os.path.exists(args.path):
                raise FileNotFoundError(f"Input path '{args.path}' does not exist.")

            cut_video(args, './raw')
            self.assertTrue(os.path.exists('./raw/intel_0.wav'))

if __name__ == "__main__":
    unittest.main()
