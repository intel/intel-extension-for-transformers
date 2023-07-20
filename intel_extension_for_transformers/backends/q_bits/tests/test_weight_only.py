import torch
import unittest
from q_bits import convert_to_quantized_model, QBitsConfig

torch.ops.load_library("../q_bits/libweight_only_jblasop.so")

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)

    def forward(self, x):
        return self.linear(x)


class TestWeightOnly(unittest.TestCase):
    def test_int4(self):
        raw_wei = torch.rand(2,3, dtype=torch.float)
        torch.ops.weight_only_jblasop.jblas_symqdq_s4weight(raw_wei,True,32)
        model = M()
        with torch.no_grad():
            model.linear.weight = torch.nn.Parameter(raw_wei)
        activation = torch.rand(1,3, dtype=torch.float)
        output = model(activation)

        config = QBitsConfig(quant_bits=4, quant_type="int4")
        convert_to_quantized_model(model, config)
        output_quant = model(activation)
        print(output)
        print(output_quant)
        assert torch.allclose(output, output_quant)


if __name__ == "__main__":
    unittest.main()
