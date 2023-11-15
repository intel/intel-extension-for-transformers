import unittest

from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModel, WeightOnlyQuantConfig, AutoModelForCausalLM
from intel_extension_for_transformers.llm.runtime.graph.scripts.convert import convert_model
from intel_extension_for_transformers.llm.runtime.graph import Model

import numpy as np
import sys

def cmpData(numa, numb):
    if (numa.shape != numb.shape):
        return 1
    totalErr = ((np.abs(numa - numb))**2).sum()
    totalNum = (np.abs(numa)**2).sum()
    return np.sqrt(totalErr/totalNum)

class TestLLMRUNTIME(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        # shutil.rmtree("./runtime_outs/ne_chatglm_q.bin", ignore_errors=True)
        # shutil.rmtree("./runtime_outs/ne_gptj_f32.bin", ignore_errors=True)
        pass

    # def test_llm_runtime(self):

    #     model_name = "/mnt/disk1/data2/zhenweil/models/chatglm2-6b"  # or local path to model
    #     woq_config = WeightOnlyQuantConfig(compute_dtype="int8", weight_dtype="int4", use_cache=True)
    #     prompt = "小明的妈妈有三个孩子，老大叫大毛，老二叫二毛，老三叫什么？"

    #     tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    #     input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    #     streamer = TextStreamer(tokenizer)

    #     model = AutoModel.from_pretrained(model_name, quantization_config=woq_config, use_llm_runtime=True, trust_remote_code=True)
    #     gen_tokens = model.generate(input_ids, streamer=streamer, max_new_tokens=300, seed=1)
    #     outputs = tokenizer.batch_decode(gen_tokens)
    #     print(outputs)
    #     self.assertTrue("小明" in outputs[0])

    # def test_beam_search(self):
    #     model_name = "/mnt/disk1/data2/zhenweil/models/gpt_j/gpt-j-6B"  # or local path to model
    #     prompts = [
    #        "she opened the door and see",
    #        "tell me 10 things about jazz music",
    #        "What is the meaning of life?",
    #        "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer"\
    #         " The slings and arrows of outrageous fortune, "\
    #         "Or to take arms against a sea of troubles."\
    #         "And by opposing end them. To die—to sleep,"
    #         ]

    #     tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,
    #                                               padding_side="left")
    #     tokenizer.pad_token = tokenizer.eos_token
    #     pad_token = tokenizer(tokenizer.pad_token)['input_ids'][0]
    #     inputs = tokenizer(prompts, padding=True, return_tensors='pt')

    #     # pytorch fp32
    #     # pt_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    #     # pt_model.eval() 
    #     # pt_generate_ids = pt_model.generate(**inputs, max_new_tokens=128, min_new_tokens=30,
    #     #                                     early_stopping=True, num_beams=4).tolist()
    #     # llm runtime fp32
    #     woq_config = WeightOnlyQuantConfig(not_quant=True, use_cache=True)
    #     itrex_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=woq_config, trust_remote_code=True)
    #     itrex_generate_ids = itrex_model.generate(inputs.input_ids, num_beams=4, # batch_size=4, 
    #                               max_new_tokens=128, min_new_tokens=30, early_stopping=True,
    #                               pad_token=pad_token)
    #     # for i in range(len(itrex_generate_ids)):
    #     #     self.assertListEqual(pt_generate_ids[i], itrex_generate_ids[i])


    def test_llm_runtime(self):

        model_name = "/mnt/disk1/data2/zhenweil/models/chatglm2-6b"  # or local path to model
        woq_config = WeightOnlyQuantConfig(compute_dtype="int8", weight_dtype="int4", use_cache=True)
        prompt = "hi, tell me about Intel"

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        inputs = tokenizer(prompt, return_tensors="pt")

        # pytorch fp32
        pt_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        pt_model.eval() 
        logits = pt_model(**inputs).logits[:,-1]

        model = AutoModel.from_pretrained(model_name, quantization_config=woq_config, use_llm_runtime=True, trust_remote_code=True)
        outputs = model.forward(inputs.input_ids)
        print(cmpData(logits.detach().numpy().flatten(), outputs.flatten()))

if __name__ == "__main__":
    unittest.main()
