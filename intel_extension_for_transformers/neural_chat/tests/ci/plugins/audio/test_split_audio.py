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

from intel_extension_for_transformers.neural_chat.pipeline.plugins.audio.utils.split_audio import main
import os
import argparse
import shlex
import shutil
import unittest

class TestSplitAudio(unittest.TestCase):
    def setUp(self):
        shutil.rmtree("./split", ignore_errors=True)
        os.mkdir("./split")

    def tearDown(self) -> None:
        shutil.rmtree("./split", ignore_errors=True)

    def test_split_audio(self):
            # Usage: split_audio.py --ag (0~3) --in_path <input path> --out_path <output path>
            parser = argparse.ArgumentParser(__doc__)
            parser.add_argument("--ag", type=int, default=3)
            audio_path = \
              "/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/assets/audio/sample.wav"
            if os.path.exists(audio_path):
                parser.add_argument("--in_path", type=str, default=audio_path)
            else:
                parser.add_argument("--in_path", type=str, default="../assets/audio/sample.wav")
            parser.add_argument("--out_path", type=str, default="./split",
                                help="please use relative path")
            parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
            args = parser.parse_args()

            is_exist = os.path.exists(args.in_path)
            if not is_exist:
                print("path not existed!")
            else:
                main(args)

            if os.path.exists(audio_path):
                audio_path_prefix = \
                    "/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/assets/audio"
                self.assertTrue(os.path.exists(audio_path_prefix + '/split/sample_00.wav'))
            else:
                self.assertTrue(os.path.exists('../assets/audio/split/sample_00.wav'))

if __name__ == "__main__":
    unittest.main()
