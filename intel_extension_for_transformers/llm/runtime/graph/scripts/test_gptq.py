import torch
import os
from transformers import AutoTokenizer, TextStreamer
from transformers import AutoModelForCausalLM
model_name = "/mnt/disk1/data2/zhenweil/models/bloom/bloom-7b1"
prompt = "Once upon a time, a little girl"

tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
from neural_compressor.utils.pytorch import load
# model = Model()
new_model = load("/mnt/disk1/data2/zhenweil/models/bloom/bloom-gptq", model, weight_only=True)


# import sys
# sys.exit()

out1 = new_model(inputs)
# self.assertTrue(torch.all(out1 == out2))

# model_size1 = os.path.getsize("saved/best_model.pt") / 1024
# print("FP32 Model size:{:.3f}M".format(model_size1))
from neural_compressor.model import Model as INCModel

inc_model = INCModel(new_model)

inc_model.export_compressed_model(qweight_config_path="/mnt/disk1/data2/zhenweil/models/bloom/bloom-gptq/qconfig.json")
import pudb; pudb.set_trace()
outs = inc_model(inputs)
print(outs)
# torch.save(inc_model.state_dict(), "saved/tmp.pt")
# model_size2 = os.path.getsize("saved/tmp.pt") / 1024
# print("WeightOnlyLinear Model size:{:.3f}M".format(model_size2))
# self.assertTrue(isinstance(inc_model.model.fc1, WeightOnlyLinear))
# self.assertTrue(model_size1 / model_size2 > 2)