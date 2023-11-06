import copy
import os
import torch
import unittest
import shutil
from intel_extension_for_transformers.transformers.modeling import AutoModelForCausalLM
from intel_extension_for_transformers.llm.quantization.nn.cpu.modules import QuantizedLinearCPU
from intel_extension_for_transformers.llm.quantization.utils import convert_to_quantized_model, replace_linear
from intel_extension_for_transformers.transformers import WeightOnlyQuantConfig


class M(torch.nn.Module):
    def __init__(self, with_bias=False):
        super().__init__()
        self.linear = torch.nn.Linear(32, 2, bias=with_bias)

    def forward(self, x):
        return self.linear(x)


llama_model_path = "fxmarty/tiny-llama-fast-tokenizer"

class TestWeightOnly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = "./woq_tmp"
        # if workspace not exist, crate it
        if not os.path.exists(cls.workspace):
            os.mkdir(cls.workspace)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.workspace, ignore_errors=True)

    def test_woq_config(self):
        config = WeightOnlyQuantConfig(weight_dtype="int4_fullrange", group_size=32)
        diff_res = config.to_diff_dict()
        ref_config = {'weight_dtype': 'int4_fullrange'}
        self.assertEqual(diff_res, ref_config)
        print(diff_res)
        print(config.to_dict())
        print(config.to_json_string())
        config.to_json_file(f"{self.workspace}/config.json")
        print(config)

    def test_int8(self):
        raw_wei = torch.rand(2, 32, dtype=torch.float)
        compress_wei = torch.ops.weight_only_jblasop.qbits_quantize(
            raw_wei, True, 32, "fp32", "s8_scalef32")
        revert_wei = torch.zeros(2, 32, dtype=torch.float)
        torch.ops.weight_only_jblasop.qbits_dequantize(
            compress_wei, revert_wei, True, "fp32", "s8_scalef32")
        for bias in [True, False]:
            model = M(with_bias=bias)
            with torch.no_grad():
                model.linear.weight = torch.nn.Parameter(revert_wei)
            activation = torch.rand(1,32, dtype=torch.float)
            output = model(activation)

            config = WeightOnlyQuantConfig(weight_dtype="int8", group_size=32)
            model = convert_to_quantized_model(model, config)
            output_quant = model(activation)
            print(output)
            print(output_quant)
            assert torch.allclose(output, output_quant, rtol=0.01)

    def test_int4(self):
        raw_wei = torch.rand(2, 32, dtype=torch.float)
        compress_wei = torch.ops.weight_only_jblasop.qbits_quantize(
            raw_wei, True, 32, "fp32", "s4fullrange_scalef32")
        revert_wei = torch.zeros(2, 32, dtype=torch.float)
        torch.ops.weight_only_jblasop.qbits_dequantize(
            compress_wei, revert_wei, True, "fp32", "s4fullrange_scalef32")
        for bias in [True, False]:
            model = M(with_bias=bias)
            with torch.no_grad():
                model.linear.weight = torch.nn.Parameter(revert_wei)
            activation = torch.rand(1, 5, 32, dtype=torch.float)
            output = model(activation)
            with torch.no_grad():
                model.linear.weight = torch.nn.Parameter(raw_wei)

            config = WeightOnlyQuantConfig(weight_dtype="int4_fullrange", group_size=32)
            model = convert_to_quantized_model(model, config)
            output_quant = model(activation)
            print(output)
            print(output_quant)
            assert torch.allclose(output, output_quant, rtol=0.01)

    # def test_int4_training(self):
    #     class LinearPredictor(torch.nn.Module):
    #         def __init__(self, *args, **kwargs) -> None:
    #             super().__init__(*args, **kwargs)
    #             self.inlinear = torch.nn.Linear(1, 64, bias=True)
    #             self.middlelinear = torch.nn.Linear(64, 128, bias=True)
    #             self.outlinear = torch.nn.Linear(128, 1, bias=True)
    #             self.classifier = torch.nn.Sigmoid()

    #         def forward(self, x):
    #             x = self.inlinear(x)
    #             x = self.middlelinear(x)
    #             x = self.outlinear(x)
    #             x = self.classifier(x)
    #             return x

    #     model = LinearPredictor()
    #     replace_linear(model, None, None, WeightOnlyQuantConfig(weight_dtype='int4_fullrange'))
    #     lossfn = torch.nn.MSELoss()
    #     optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    #     batch_size = 16
    #     for i in range(200):
    #         x = torch.randn((batch_size,1))
    #         out = model(x)
    #         loss = lossfn(out, (x>=0).float())
    #         loss.backward()
    #         optimizer.step()
    #         if (i+1) % 50 == 0:
    #             print(f"Step:{i+1}, Loss:{loss.item()}")

    #     x = torch.randn((batch_size, 1)) / 2
    #     out = model(x)
    #     accuracy = ((out>=0.5).float() == (x>=0).float()).sum() / batch_size * 100
    #     print("Accuracy:{:.2f}%".format(accuracy))
    #     self.assertTrue(accuracy > 90)

    def test_auto_model(self):
        model = AutoModelForCausalLM.from_pretrained(llama_model_path, load_in_4bit=True, use_llm_runtime=False)
        module_list = []
        for name, module in model.named_modules():
            if isinstance(module, QuantizedLinearCPU):
                module_list.append(name)
        self.assertTrue(len(module_list) > 0)

    def test_auto_model_with_config(self):
        config = WeightOnlyQuantConfig()
        model = AutoModelForCausalLM.from_pretrained(
            llama_model_path, quantization_config=config, use_llm_runtime=False
        )
        module_list = []
        for name, module in model.named_modules():
            if isinstance(module, QuantizedLinearCPU):
                module_list.append(name)
        self.assertTrue(len(module_list) > 0)

    def test_auto_model_saving_loading(self):
        model = AutoModelForCausalLM.from_pretrained(llama_model_path, load_in_4bit=True, use_llm_runtime=False)
        module_list = []
        for name, module in model.named_modules():
            if isinstance(module, QuantizedLinearCPU):
                module_list.append(name)
        self.assertTrue(len(module_list) > 0)
        model.save_low_bit(self.workspace)
        loaded_model = AutoModelForCausalLM.load_low_bit(self.workspace)
        for name, module in loaded_model.named_modules():
            if isinstance(module, QuantizedLinearCPU):
                module_list.append(name)
        self.assertTrue(len(module_list) > 0)


if __name__ == "__main__":
    unittest.main()
