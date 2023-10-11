import torch
import os
from transformers import AutoTokenizer, TextStreamer
from transformers import AutoModelForCausalLM
model_name = "/mnt/disk1/data2/zhenweil/models/llama/Llama-2-7b-chat-hf"
prompt = "Once upon a time, a little girl"

tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
from neural_compressor.utils.pytorch import load
# model = Model()
# new_model = load("/mnt/disk1/data2/zhenweil/models/llama/llama2-gptq", model, weight_only=True)
# out1 = new_model(inputs)
# from neural_compressor.model import Model as INCModel
# inc_model = INCModel(new_model)
# inc_model.export_compressed_model(qweight_config_path="/mnt/disk1/data2/zhenweil/models/llama/llama2-gptq/qconfig.json",
#                                   gptq_config_path="/mnt/disk1/data2/zhenweil/models/llama/llama2-gptq/gptq_config.json")

print("export_compressed_model done")
gen_tokens = model.generate(inputs, streamer=streamer, max_new_tokens=300)
outputs = tokenizer.batch_decode(gen_tokens)