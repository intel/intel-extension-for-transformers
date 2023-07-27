import copy
import numpy as np
import torch
import unittest
from q_bits import convert_to_quantized_model, QBitsConfig
from q_bits.utils import replace_linear


class M(torch.nn.Module):
    def __init__(self, with_bias=False):
        super().__init__()
        self.linear = torch.nn.Linear(32, 2, bias=with_bias)

    def forward(self, x):
        return self.linear(x)


class TestWeightOnly(unittest.TestCase):
    def test_int8(self):
        raw_wei = torch.rand(2, 32, dtype=torch.float)
        torch.ops.weight_only_jblasop.jblas_symqdq_weight(raw_wei, True, 8, 32)
        for bias in [True, False]:
            model = M(with_bias=bias)
            with torch.no_grad():
                model.linear.weight = torch.nn.Parameter(raw_wei)
            activation = torch.rand(1,32, dtype=torch.float)
            output = model(activation)

            config = QBitsConfig(quant_bits=8, quant_type="int8", group_size=32)
            convert_to_quantized_model(model, config)
            output_quant = model(activation)
            print(output)
            print(output_quant)
            assert torch.allclose(output, output_quant, rtol=0.01)

    def test_int4(self):
        raw_wei = torch.rand(2, 32, dtype=torch.float)
        for bias in [True, False]:
            model = M(with_bias=bias)
            fake_quant_wei = copy.deepcopy(raw_wei)
            torch.ops.weight_only_jblasop.jblas_symqdq_weight(fake_quant_wei, True, 4, 32)
            with torch.no_grad():
                model.linear.weight = torch.nn.Parameter(fake_quant_wei)
            activation = torch.rand(1,32, dtype=torch.float)
            output = model(activation)
            with torch.no_grad():
                model.linear.weight = torch.nn.Parameter(raw_wei)

            config = QBitsConfig(quant_bits=4, quant_type="int4", group_size=32)
            convert_to_quantized_model(model, config)
            output_quant = model(activation)
            print(output)
            print(output_quant)
            assert torch.allclose(output, output_quant, rtol=0.01)

    def test_int4_training(self):
        class LinearPredictor(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.inlinear = torch.nn.Linear(1, 64, bias=True)
                self.middlelinear = torch.nn.Linear(64, 128, bias=True)
                self.outlinear = torch.nn.Linear(128, 1, bias=True)
                self.classifier = torch.nn.Sigmoid()

            def forward(self, x):
                x = self.inlinear(x)
                x = self.middlelinear(x)
                x = self.outlinear(x)
                x = self.classifier(x)
                return x

        model = LinearPredictor()
        replace_linear(model, None, None, QBitsConfig(4, quant_type='int4'))
        lossfn = torch.nn.MSELoss()
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        batch_size = 16
        for i in range(200):
            x = torch.randn((batch_size,1))
            out = model(x)
            loss = lossfn(out, (x>=0).float())
            loss.backward()
            optimizer.step()
            if (i+1) % 50 == 0:
                print(f"Step:{i+1}, Loss:{loss.item()}")

        x = torch.randn((batch_size, 1)) / 2
        out = model(x)
        accuracy = ((out>=0.5).float() == (x>=0).float()).sum() / batch_size * 100
        print("Accuracy:{:.2f}%".format(accuracy))
        self.assertTrue(accuracy > 90)


if __name__ == "__main__":
    unittest.main()
