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

import os
import shutil
import unittest
from unittest.mock import MagicMock
import torch

class TestDevice(unittest.TestCase):
    def test_cpu(self):
        torch.cuda.is_available = MagicMock(return_value=False)
        from intel_extension_for_transformers.utils.device_utils import is_hpu_available, get_device_type
        device = get_device_type()
        self.assertTrue("cpu" in device)

    def test_gpu(self):
        torch.cuda.is_available = MagicMock(return_value=True)
        from intel_extension_for_transformers.utils.device_utils import is_hpu_available, get_device_type
        device = get_device_type()
        self.assertTrue("cuda" in device)


if __name__ == "__main__":
    unittest.main()
