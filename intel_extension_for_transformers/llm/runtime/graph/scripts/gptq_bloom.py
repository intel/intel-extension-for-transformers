from transformers import AutoTokenizer, TextStreamer, AutoConfig
from transformers import AutoModelForCausalLM
import json

model_name = "/mnt/disk1/data2/zhenweil/models/bloom/bloom-7b1"
prompt = "Once upon a time, a little girl"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

ori_out = model(inputs).logits
# import pudb; pudb.set_trace()

gptq_model = "/mnt/disk1/data2/zhenweil/models/bloom/bloom-gptq/"
from neural_compressor.utils.pytorch import load
# import pudb; pudb.set_trace()
new_model = load(gptq_model, model, weight_only=True)
# import pdb; pdb.set_trace()
from neural_compressor.model import Model as INCModel
inc_model = INCModel(new_model)
inc_model.export_compressed_model(
    qweight_config_path=gptq_model+"/qconfig.json",
    gptq_config_path=gptq_model+"/gptq_config.json"
)

# new_out = inc_model(inputs).logits
# print((new_out - ori_out).abs().sum())
# outputs = tokenizer.batch_decode(gen_tokens)
# print(outputs)
print("export_compressed_model done")
gen_tokens = new_model.generate(inputs, streamer=streamer, max_new_tokens=300)
outputs = tokenizer.batch_decode(gen_tokens)
print(outputs)