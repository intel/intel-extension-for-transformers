# Streaming LLM
## Introduction
The Intel Extension for Transformers  has identified specific issues that the LLM may encounter in the Chat Scene:
- Limited Output Length: The LLM model is primarily pretrained on a limited sequence length. Consequently, its accuracy diminishes when the sequence length exceeds the attention window size used in pretraining.
- Inefficiency: In the decoding phase, Transformer-based LLMs store Key and Value states (KV) for all prior tokens, resulting in excessive memory usage and higher decoding latency.

To address this problem, we integrate Streaming LLM into Intel Extension for Transformers, bringing about substantial improvements in memory usage. Unlike the traditional KV cache algorithms, our approach incorporates Attention Sink (four initial tokens) to stabilize attention computation, and the Rolling KV Cache retains the most recent tokens, crucial for language modeling. This design is remarkably flexible, seamlessly integrating into autoregressive language models that leverage relative positional encoding, such as RoPE and ALiBi.
## Example
```python
from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM, RtnConfig
model_name = "Intel/neural-chat-7b-v1-1"     # Hugging Face model_id or local model
rtn_config = RtnConfig(compute_dtype="int8", weight_dtype="int4")
prompt = "Once upon a time, a little girl"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=rtn_config, trust_remote_code=True)
 
# Recommend n_keep=4 to do attention sinks (four initial tokens) and n_discard=-1 to drop half rencetly tokens when meet length threshold
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300, ctx_size=100, n_keep=4, n_discard=-1)
```
please refer [here](https://github.com/intel/neural-speed/blob/main/docs/supported_models.md) for more supported models and refer [here](https://medium.com/intel-analytics-software/efficient-streaming-llm-with-intel-extension-for-transformers-runtime-31ee24577d26) for more details.
