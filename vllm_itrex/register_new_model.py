from vllm import LLM
from vllm import ModelRegistry

from vllm_chatglm_model import ChatGLMForCausalLM, ChatGLMModel
#from transformer_chatglm_model import ChatGLMForConditionalGeneration

ModelRegistry.register_model("ChatGLMModel", ChatGLMForCausalLM)            #这样可以
#ModelRegistry.register_model("ChatGLMModel_XZZ", ChatGLMForCausalLM)        #这样不行


prompts = ["你好"]  # Sample prompts.
llm = LLM(model="/home/zhenzhong/model/chatglm2-6b", trust_remote_code=True)  # Create an LLM.


outputs = llm.generate(prompts)  # Generate texts from the prompts.
print(outputs[0])
