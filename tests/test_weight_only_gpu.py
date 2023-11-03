import os
import torch
import unittest
import shutil
from intel_extension_for_transformers.transformers.modeling import AutoModelForCausalLM
from intel_extension_for_transformers.transformers import WeightOnlyQuantConfig
from transformers import AutoTokenizer
from intel_extension_for_transformers.utils.utils import get_gpu_family
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.nn.functional as F


class DummyDataset(data.Dataset):
    def __init__(self, model_name, seqlen):
        self.seqlen = seqlen
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            model_max_length=self.seqlen,
            padding_side="right",)
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
        name = get_gpu_family()
        if name != 'max':
            print("There is no PVC (Max) GPU, skip this function {}".format('test_int4_ipex_pvc'))
            return True

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
        name = get_gpu_family()
        if name != 'max':
            print("There is no PVC (Max) GPU, skip this function {}".format('test_int4_ipex_pvc'))
            return True

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


    def test_int4_ipex_arc(self):
        name = get_gpu_family()
        if name != 'arc':
            print("There is no Arc GPU, skip this function {}".format('test_int4_ipex_arc'))
            return True

        from intel_extension_for_transformers.llm.quantization.utils import convert_to_quantized_model
        import intel_extension_for_pytorch as ipex

        device_map = "xpu"

        model_name ="fxmarty/tiny-llama-fast-tokenizer"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float)
        model.seqlen = 2048
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        prompt = "how to test the code?"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        output = model(input_ids)

        config = WeightOnlyQuantConfig(weight_dtype="int4_fullrange", group_size=16)
        config.calib_dataloader = DataLoader(
            DummyDataset(model_name, model.seqlen),
            batch_size=1,
            shuffle=False,
        )
        qmodel = convert_to_quantized_model(model, config, device=torch.device(device_map))
        output_quant = qmodel(input_ids.to(torch.device("xpu")))
        fp16_logits = output['logits']
        quan_logits = output_quant['logits'].to('cpu')
        print("fp16 logits {}".format(fp16_logits.shape))
        print("int4 logits {}".format(quan_logits.shape))

        return True

if __name__ == "__main__":
    unittest.main()
