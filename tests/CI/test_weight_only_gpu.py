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
            res = torch.tensor(input_ids), torch.tensor(self.encoded_dict['attention_mask'])
            return res


class M(torch.nn.Module):
    def __init__(self, with_bias=False):
        super().__init__()
        self.linear = torch.nn.Linear(32, 2, bias=with_bias)

    def forward(self, x):
        return self.linear(x)


@unittest.skipIf(not _ipex_available or gpu_name == "no_gpu",
    "There is no Intel GPU in this machine, skip this test!")
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

        model_name ="hf-internal-testing/tiny-random-gptj"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        prompt = "how to test the code?"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device_map)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=device_map)
        model.seqlen = 2048
        output = model.generate(input_ids)
        fp16_out = output.to("cpu")
        print("fp16 logits {}".format(fp16_out.shape))

        config = WeightOnlyQuantConfig(weight_dtype="int4_fullrange",
                                       group_size=32,
                                       compute_dtype="fp16",
                                       scale_dtype="fp16")
        config.calib_dataloader = DataLoader(
            DummyDataset(model_name, model.seqlen),
            batch_size=1,
            shuffle=False,
        )
        qmodel = AutoModelForCausalLM.from_pretrained(model_name, use_llm_runtime=False,
                                                      device_map=device_map, quantization_config=config,
                                                      trust_remote_code=True, torch_dtype=torch.float16)
        qmodel.save_pretrained(self.workspace)
        # qmodel = ipex.optimize_transformers(qmodel, inplace=True, dtype=torch.float16, woq=True, device=device_map)
        output_quant = qmodel.generate(input_ids.to(torch.device(device_map)))
        quan_out = output_quant.to('cpu')
        print("int4 logits {}".format(quan_out.shape))

        # move model to CPU
        qmodel.to("cpu")
        loaded_model = AutoModelForCausalLM.from_pretrained(
            self.workspace, trust_remote_code=True, device_map=device_map, torch_dtype=torch.float16
        )
        # loaded_model = ipex.optimize_transformers(qmodel, inplace=True, dtype=torch.float16, woq=True, device=device_map)
        output_reload = loaded_model.generate(input_ids.to(torch.device(device_map)))
        reload_out = output_reload.to('cpu')
        print(fp16_out)
        print(quan_out)
        print(reload_out)
        print("!!!!!!!!!!!!", torch.max(torch.abs(quan_out - reload_out)))
        assert torch.allclose(reload_out, quan_out, rtol=0.03)


if __name__ == "__main__":
    unittest.main()
