from vllm import LLM
from torch.nn import LayerNorm
import torch
import copy

from neural_compressor import quantization
from intel_extension_for_transformers.transformers.utils import RtnConfig
from intel_extension_for_transformers.transformers.llm.quantization.utils import convert_to_quantized_model


# 为什么一定要放在上面？？
original_linear = torch.nn.Linear(in_features=4096,out_features=4608, bias=True, dtype=torch.bfloat16)
# original_linear = torch.nn.Linear(in_features=4096,out_features=4608, bias=True, dtype=torch.float32)

# nn.Linear
#prompts = ["Hello, my name is", "The capital of France is", "你好"]  # Sample prompts.
#prompts = ["hello, my name is"]
prompts = ["你好"]
llm = LLM(model="/home/zhenzhong/model/chatglm2-6b", trust_remote_code=True)  # Create an LLM.


from vllm.model_executor.model_loader import get_model, get_model_loader, get_model_architecture

model = llm.llm_engine.model_executor.driver_worker.model_runner.model
print("Original model -------------------------------------------------------------------------------- ", model)

# replace RMSNorM by Layernorm
# layer_norm_func = LayerNorm(4096, eps=1e-5, dtype=torch.bfloat16)
# for idx in range(28):
#     print("modified!!!-----------------", idx)
#     model._modules['transformer']._modules['encoder']._modules['layers']._modules[str(idx)]._modules['input_layernorm'] = copy.deepcopy(layer_norm_func)
#     model._modules['transformer']._modules['encoder']._modules['layers']._modules[str(idx)]._modules['post_attention_layernorm'] = copy.deepcopy(layer_norm_func)

# model._modules['transformer']._modules['encoder']._modules['final_layernorm'] = copy.deepcopy(layer_norm_func)
# print("modified model ---------------------------------------------------------------------------------= ", model)



for idx in range(28):
    print("modified!!!-----------------", idx)
    model._modules['transformer']._modules['encoder']._modules['layers']._modules[str(idx)]._modules['self_attention']._modules['query_key_value'] = copy.deepcopy(original_linear)    
print("modified model ---------------------------------------------------------------------------------= ", model)

# load weights
loader = get_model_loader(llm.llm_engine.load_config)
model.load_weights(loader._get_weights_iterator(llm.llm_engine.model_config.model, llm.llm_engine.model_config.revision, fall_back_to_pt=True))



config = RtnConfig(compute_dtype="fp32", group_size=128, scale_dtype="fp32", weight_dtype="int4_clip", bits=4)

model = convert_to_quantized_model(model, config)

# inc_model = quantization.fit(model, config)

###
# inc_model = quantization.fit(
#     model, conf, calib_func=calib_func, calib_dataloader=calib_dataloader
# )



# import pdb;pdb.set_trace()

# inference
import time
T1 = time.time()

outputs = llm.generate(prompts)  # Generate texts from the prompts.

T2 = time.time()
print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))
print(outputs)



# model = get_model(
#             model_config=llm.llm_engine.model_config,
#             load_config=llm.llm_engine.load_config,
#             device_config=llm.llm_engine.device_config,
#             vision_language_config=llm.llm_engine.vision_language_config,
#             lora_config=llm.llm_engine.lora_config,
#             parallel_config=llm.llm_engine.parallel_config,
#             scheduler_config=llm.llm_engine.scheduler_config)

# 替换所有的dense_h_to_4h
# idx = 0
# tmp_module = None
# for name, module in model.named_modules():
#     #print(name, " -> ", module)
#     if name == "transformer.encoder.layers.0.post_attention_layernorm":
#         tmp_module = module
#     ## import pdb;pdb.set_trace()
#     # if name == "transformer":
#     #     model._modules['transformer'] = tmp_module
#     #if name == "transformer.encoder.layers." + str(idx) + ".mlp.dense_h_to_4h":
#     if name == "transformer.encoder.layers." + str(idx) + ".mlp.dense_h_to_4h":
#         #del model._modules['transformer']._modules['encoder']._modules['layers']._modules[str(idx)]._modules['mlp']._modules['dense_h_to_4h']
#         #del model._modules['transformer']._modules['encoder']._modules['layers']._modules[str(idx)]._modules['mlp']._modules['dense_4h_to_h']
#         #model._modules['transformer']._modules['encoder']._modules['layers']._modules[str(idx)]._modules['mlp']._modules['dense_h_to_4h'] = tmp_module
#         ## import pdb;pdb.set_trace()
#         idx = idx + 1