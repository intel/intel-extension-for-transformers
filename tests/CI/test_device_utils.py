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
