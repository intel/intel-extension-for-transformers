import torch
import unittest
from ..bits_quantization import convert_to_quantized_model, QBitsConfig


class TestWeightOnly(unittest.TestCase):
    def test_int4(self):
        torch.ops.load_library("../cscr/build/libweight_only_jblasop.so")
        raw_wei = torch.rand(2,3, dtype=torch.float)
        quant_wei = torch.ops.weight_only_jblasop.jblas_quantize(raw_wei, True, 8, "sym", 32, "int8")
        # fake_weight = 
        linear = torch.nn.linear(3, 2)
        with torch.no_grad():
            linear.weight = torch.nn.Parameter(raw_wei)
        activation = torch.rand(1,3, dtype=torch.float)
        output = linear(activation)

        config = QBitsConfig()
        convert_to_quantized_model(linear, config)
        output_quant = linear(activation)
        print(output)
        print(output_quant)
        assert torch.allclose(output, output_quant)

