from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM, WeightOnlyQuantConfig

# Please change to local path to model, llama2 does not support online conversion, currently.
model_name = "/home/dingyi/Llama-2-7b-chat-hf"

woq_config = WeightOnlyQuantConfig(compute_dtype="int8", weight_dtype="int4", use_cache=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
streamer = TextStreamer(tokenizer)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=woq_config, trust_remote_code=True)

while True:
    prompt = input("> ").strip()
    if prompt == "quit":
        break
    b_prompt = "[INST]{}[/INST]".format(prompt)  # prompt template for llama2
    inputs = tokenizer(b_prompt, return_tensors="pt").input_ids
    outputs = model.generate(inputs, streamer=streamer, interactive=True, ignore_prompt=True,
                num_beams=1, max_new_tokens=-1, ctx_size = 128, do_sample=True, threads=28, repetition_penalty=1.1)
