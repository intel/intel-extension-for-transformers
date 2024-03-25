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

import unittest
from unittest.mock import patch
from intel_extension_for_transformers.tools.llm_carbon_calc import main


class TestLLMCarbonCalculator(unittest.TestCase):
    def test_calc_with_inference_time(self):
        with patch("sys.argv", ["main", "-m", "27412.98", "--tdp", "350", "-c", "0.56", "-t", "6510.3"]):
            ret = main()
        assert ret == 0.0003575115963544682

    def test_calc_with_token_latency(self):
        with patch("sys.argv", ["main", "-m", "27412.98", "--tdp", "350", "-c", "0.56", "--fl", "2284.75", "--nl", "136.31", "-n", "64"]):
            ret = main()
        assert ret == 0.0005970487041784186

    def test_calc_with_missing_arg(self):
        with patch("sys.argv", ["main", "-m", "27412.98", "--tdp", "350", "-c", "0.56", "--fl", "2284.75", "-n", "64"]):
            ret = main()
        assert ret == 0.0

if __name__ == "__main__":
    unittest.main()
