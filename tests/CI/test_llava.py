import os
import unittest
from intel_extension_for_transformers.transformers.modeling.llava_models import LlavaMistralForCausalLM
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoTokenizer
import torch

os.environ["WANDB_DISABLED"] = "true"
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "true"
MODEL_NAME = "HuggingFaceM4/tiny-random-MistralForCausalLM"


class TestLLaVA(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = LlavaMistralForCausalLM.from_pretrained(
            MODEL_NAME, low_cpu_mem_usage=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.dummpy_input = {
                "input_ids": torch.tensor([[1,1,2,2]]),
                "labels": torch.tensor([[1,1,2,2]]),
                "attention_mask": torch.tensor([[1,1,1,1]]),
                "images": torch.randn([1, 3, 336, 336])
        }

    def test_forward(self):
        output = self.model(**self.dummpy_input)
        self.assertTrue(isinstance(output["loss"], torch.Tensor))

    def test_input(self):
        tmp = self.model.prepare_inputs_for_generation(**self.dummpy_input)
        self.assertTrue(isinstance(tmp["input_ids"], torch.Tensor))

    def test_init(self):
        class TestArgs:
            vision_tower = False
            mm_vision_select_layer = -2
            mm_vision_select_feature = "patch"
            pretrain_mm_mlp_adapter = None
            vision_tower = "hf-internal-testing/tiny-random-clip"
            mm_use_im_patch_token = False
            mm_use_im_start_end = False

        model_args = TestArgs()
        self.model.model.initialize_vision_modules(model_args)
        self.assertTrue(isinstance(self.model, type(self.model)))
        self.model.initialize_vision_tokenizer(model_args, self.tokenizer)
        self.assertTrue(isinstance(self.tokenizer, type(self.tokenizer)))

        image_dummpy = torch.randn([1, 3, 30, 30])

        image_feature = self.model.encode_images(image_dummpy)
        self.assertTrue(image_feature.shape == torch.Size([1, 225, 16]))


if __name__ == "__main__":
    unittest.main()
