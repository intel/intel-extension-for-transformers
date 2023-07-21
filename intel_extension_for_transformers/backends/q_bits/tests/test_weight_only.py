import numpy as np
import torch
import unittest
from q_bits import convert_to_quantized_model, QBitsConfig


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.linear(x)


class TestWeightOnly(unittest.TestCase):
    def test_int4(self):
        raw_wei = torch.rand(2, 32, dtype=torch.float)
        torch.ops.weight_only_jblasop.jblas_symqdq_weight(raw_wei, True, 8, 32)
        raw_wei.reshape(32, 2)
        raw_wei.transpose(0, 1)

        model = M()
        with torch.no_grad():
            model.linear.weight = torch.nn.Parameter(raw_wei)
        activation = torch.rand(1,32, dtype=torch.float)
        output = model(activation)

        config = QBitsConfig(quant_bits=8, quant_type="int8", group_size=32)
        convert_to_quantized_model(model, config)
        output_quant = model(activation)
        print(output)
        print(output_quant)
        assert torch.allclose(output, output_quant, rtol=0.1)


if __name__ == "__main__":
    unittest.main()
