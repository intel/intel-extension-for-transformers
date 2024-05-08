from vllm import LLM
from vllm import ModelRegistry

from vllm_chatglm_model import ChatGLMForCausalLM

ModelRegistry.register_model("ChatGLMForConditionalGeneration", ChatGLMForCausalLM)

from vllm import LLM

prompts = ["你好"]  # Sample prompts.
llm = LLM(model="/home/zhenzhong/model/chatglm2-6b", trust_remote_code=True)  # Create an LLM.

outputs = llm.generate(prompts)  # Generate texts from the prompts.
print(outputs[0])


# from new_model_baichuan import BaichuanForCausalLM
#from new_model_chatglm import ChatGLMForConditionalGeneration
#from original_baichuan import BaichuanForCausalLM
# from transformers.models.opt

# ModelRegistry.register_model("BaichuanForCausalLM", BaichuanForCausalLM)
#ModelRegistry.register_model("ChatGLMModel", ChatGLMForCausalLM) # wrong