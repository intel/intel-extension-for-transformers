import os
import torch
import unittest
import shutil
from intel_extension_for_transformers.transformers.modeling import AutoModelForCausalLM
from intel_extension_for_transformers.transformers import WeightOnlyQuantConfig
from transformers import AutoTokenizer
from intel_extension_for_transformers.utils.utils import get_gpu_family, _ipex_available
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.nn.functional as F


if _ipex_available:
    gpu_name = get_gpu_family()


def dequantize(weight, scales, zero_points, group_size, gemm_num=1):
    k = weight.size()[-2]
    n = weight.size()[-1]
    weight = weight.reshape([gemm_num, k, n])
    n = n * 2
    group_num = int(k / group_size)
    scales = scales.reshape([gemm_num, group_num, n])
    zero_points = zero_points.reshape([gemm_num, group_num, int(n / 2)])
    weight_even = (weight & 0x0F).to(torch.int8)
    weight_odd = (weight >> 4).to(torch.int8)
    zp_even = (zero_points & 0x0F).to(torch.int8)
    zp_even += 1
    zp_odd = (zero_points >> 4).to(torch.int8)
    zp_odd += 1
    weight_fp16 = []
    zp_fp16 = []
    for ind in range(0, n):
        if ind % 2 == 0:
            weight_fp16.append(
                weight_even[:, :, int(ind / 2)].reshape([gemm_num, k, 1])
            )
            zp_fp16.append(
                zp_even[:, :, int(ind / 2)].reshape([gemm_num, group_num, 1])
            )
        else:
            weight_fp16.append(
                weight_odd[:, :, int(ind / 2)].reshape([gemm_num, k, 1])
            )
            zp_fp16.append(
                zp_odd[:, :, int(ind / 2)].reshape([gemm_num, group_num, 1])
            )
    weight_fp16 = torch.concat(weight_fp16, dim=2)
    zp_fp16 = torch.concat(zp_fp16, dim=2)
    scales = torch.reshape(scales, [gemm_num, group_num, 1, n])
    zp_fp16 = torch.reshape(zp_fp16, [gemm_num, group_num, 1, n])
    scales = scales.repeat([1, 1, group_size, 1])
    zp_fp16 = zp_fp16.repeat([1, 1, group_size, 1])
    scales = torch.reshape(scales, [gemm_num, k, n])
    zp_fp16 = torch.reshape(zp_fp16, [gemm_num, k, n])
    # weight_fp16 = ((weight_fp16 - zp_fp16).to(torch.float16)) * scales
    weight_fp16 = ((weight_fp16 - 8 ).to(torch.float16)) * scales
    if gemm_num == 1:
        weight_fp16 = weight_fp16.reshape([k, n])
    return weight_fp16


class DummyDataset(data.Dataset):
    def __init__(self, model_name, seqlen):
        self.seqlen = seqlen
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            model_max_length=self.seqlen,
            padding_side="right",
            trust_remote_code=True)
        self.sequence_a = "intel-extension-for-transformers is based in SH"
        self.sequence_b = "Where is intel-extension-for-transformers based? NYC or SH"
        self.encoded_dict = self.tokenizer(self.sequence_a, self.sequence_b)
        self.encoded_dict['labels'] = 1


    def __len__(self):
        return 10

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        if index < 10:
            input_ids = torch.tensor(self.encoded_dict['input_ids'])
            input_len = input_ids.shape[-1]
            attention_mask = self.encoded_dict['attention_mask']
            pad_size = self.seqlen - input_len
            input_ids = F.pad(input_ids, pad=(0, pad_size), value=0)
            res = torch.tensor(input_ids),torch.tensor(self.encoded_dict['attention_mask'])
            return res


class M(torch.nn.Module):
    def __init__(self, with_bias=False):
        super().__init__()
        self.linear = torch.nn.Linear(32, 2, bias=with_bias)

    def forward(self, x):
        return self.linear(x)


@unittest.skipIf(not _ipex_available or gpu_name != "max",
    "There is no PVC(Max) GPU in this machine, skip this test!")
class TestWeightOnly(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.workspace = "./woq_config_ipex_tmp"
        # if workspace not exist, create it
        if not os.path.exists(cls.workspace):
            os.mkdir(cls.workspace)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.workspace, ignore_errors=True)

    def test_int4_ipex_pvc(self):
        from intel_extension_for_transformers.llm.quantization.utils import convert_to_quantized_model_by_ipex
        import intel_extension_for_pytorch as ipex

        device_map = "xpu"

        model_name ="EleutherAI/gpt-j-6B"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float)
        model.seqlen = 2048
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        prompt = "how to test the code?"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        output = model(input_ids)

        config = WeightOnlyQuantConfig(weight_dtype="int4_fullrange", group_size=4096)
        config.calib_dataloader = DataLoader(
            DummyDataset(model_name, model.seqlen),
            batch_size=1,
            shuffle=False,
        )
        qmodel = convert_to_quantized_model_by_ipex(model, config, device=torch.device(device_map))
        output_quant = qmodel(input_ids.to(torch.device("xpu")))
        fp16_logits = output['logits']
        quan_logits = output_quant['logits'].to('cpu')
        print("fp16 logits {}".format(fp16_logits.shape))
        print("int4 logits {}".format(quan_logits.shape))

        return True

    def test_save_load_int4_ipex_pvc(self):
        from intel_extension_for_transformers.llm.quantization.utils import convert_to_quantized_model_by_ipex
        import intel_extension_for_pytorch as ipex

        device_map = "xpu"
        seqlen = 2048
        model_name ="EleutherAI/gpt-j-6B"
        config = WeightOnlyQuantConfig(weight_dtype="int4_fullrange", group_size=4096)
        config.calib_dataloader = DataLoader(
            DummyDataset(model_name, seqlen),
            batch_size=1,
            shuffle=False,
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, use_llm_runtime=False, device_map = device_map, \
                                                        quantization_config = config)
        model.save_low_bit(self.workspace)
        model = None
        loaded_model = AutoModelForCausalLM.load_low_bit(self.workspace)
        module_list = []
        # QuantizedLinearGPU_PVC = ipex.nn.optimize_transformers.modules.Layers.IpexFastLinear
        QuantizedLinearGPU_PVC = ipex.nn.utils._quantize_convert.INT4Linear
        for name, module in loaded_model.named_modules():
            if isinstance(module, QuantizedLinearGPU_PVC):
                module_list.append(name)
        self.assertTrue(len(module_list) > 0)

        return True


@unittest.skipIf(not _ipex_available or gpu_name != "arc",
    "There is no ARC GPU in this machine, skip this test!")
class TestArcWeightOnly(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.workspace = "./woq_config_ipex_tmp"
        # if workspace not exist, create it
        if not os.path.exists(cls.workspace):
            os.mkdir(cls.workspace)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.workspace, ignore_errors=True)

    def test_int4_ipex_arc_with_auto(self):
        import intel_extension_for_pytorch as ipex

        device_map = "xpu"

        # model_name ="hf-internal-testing/tiny-random-gptj"
        model_name ="Qwen/Qwen-7B-Chat"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        prompt = "how to test the code?"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device_map)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            # fp16=True,
            torch_dtype=torch.float16,
            device_map=device_map)
        model.to(device_map)
        model.seqlen = 2048
        output = model(input_ids)
        fp16_logits = output['logits'].to("cpu")
        print("fp32 logits {}".format(fp16_logits.shape))

        config = WeightOnlyQuantConfig(weight_dtype="int4_fullrange",
                                       group_size=128,
                                       compute_dtype="fp16",
                                       scale_dtype="fp16")
        config.calib_dataloader = DataLoader(
            # DummyDataset(model_name, model.seqlen),
            DummyDataset(model_name, 2048),
            batch_size=1,
            shuffle=False,
        )
        qmodel = AutoModelForCausalLM.from_pretrained(model_name, use_llm_runtime=False,
                                                      device_map=device_map, quantization_config=config,
                                                      trust_remote_code=True, fp16=True)
        qmodel.save_low_bit(self.workspace)
        # qmodel = ipex.optimize_transformers(qmodel, inplace=True)
        output_quant = qmodel(input_ids.to(torch.device("xpu")))
        quan_logits = output_quant['logits'].to('cpu')
        print("int4 logits {}".format(quan_logits.shape))

        # move model to CPU
        qmodel.to("cpu")
        loaded_model = AutoModelForCausalLM.load_low_bit(self.workspace, trust_remote_code=True)
        output_reload = loaded_model(input_ids.to(torch.device("xpu")))
        reload_logits = output_reload['logits'].to('cpu')
        print(quan_logits)
        print(reload_logits)
        print("!!!!!!!!!!!!", torch.max(torch.abs(quan_logits - reload_logits)))
        assert torch.allclose(reload_logits, quan_logits, rtol=0.03)

    def test_int4_ipex_arc(self):
        from intel_extension_for_transformers.llm.quantization.utils import convert_to_quantized_model
        import intel_extension_for_pytorch as ipex
        for bias in [True, False]:
            model = M(with_bias=bias).to(dtype=torch.float16)
            activation = torch.rand(1, 32, dtype=torch.float16)
            output = model(activation)

            config = WeightOnlyQuantConfig(weight_dtype="int4_fullrange", group_size=32, compute_dtype="fp16", scale_dtype="fp16")
            model = convert_to_quantized_model(model, config, device="xpu")
            # model = ipex.optimize_transformers(model, inplace=True)
            output_quant = model(activation.to(torch.device("xpu")))
            print(output)
            print(output_quant)
            assert torch.allclose(output, output_quant.to("cpu"), rtol=0.03)


if __name__ == "__main__":
    unittest.main()
