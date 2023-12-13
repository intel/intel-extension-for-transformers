Weight Only Quantization (WOQ)
=====

1. [Introduction](#introduction)

2. [Supported Framework Model Matrix](#supported-framework-model-matrix)

3. [Examples](#examples)

## Introduction

As large language models (LLMs) become more prevalent, there is a growing need for new and improved quantization methods that can meet the computational demands of these modern architectures while maintaining the accuracy. Compared to [normal quantization](https://github.com/intel/intel-extension-for-transformers/blob/main/docs/quantization.md) like W8A8, weight only quantization is probably a better trade-off to balance the performance and the accuracy, since we will see below that the bottleneck of deploying LLMs is the memory bandwidth and normally weight only quantization could lead to better accuracy.
## Supported Framework Model Matrix

| Algorithms/Framework |   PyTorch  |    LLM Runtime    |
|:--------------:|:----------:|:----------:|
|       RTN      |  &#10004;  |  &#10004;  |
|       AWQ      |  &#10004;  | stay tuned |
|      TEQ      | &#10004; | stay tuned |
|      GPTQ      | stay tuned | &#10004; |
> **RTN:** A quantification method that we can think of very intuitively. It does not require additional datasets and is a very fast quantization method. Generally speaking, RTN will convert the weight into a uniformly distributed integer data type, but some algorithms, such as Qlora, propose a non-uniform NF4 data type and prove its theoretical optimality.

> **GPTQ:** A new one-shot weight quantization method based on approximate second-order information, that is both highly-accurate and highly efficient. The weights of each column are updated based on the fixed-scale pseudo-quantization error and the inverse of the Hessian matrix calculated from the activations. The updated columns sharing the same scale may generate a new max/min value, so the scale needs to be saved for restoration.

> **AWQ:** Proved that protecting only 1% of salient weights can greatly reduce quantization error. the salient weight channels are selected by observing the distribution of activation and weight per channel. The salient weights are also quantized after multiplying a big scale factor before quantization for preserving. 

> **TEQ:** A trainable equivalent transformation that preserves the FP32 precision in weight-only quantization. It is inspired by AWQ while providing a new solution to search for the optimal per-channel scaling factor between activations and weights.


## Examples

Our motivation is improve CPU support for weight only quantization, since `bitsandbytes` only support CUDA GPU device. We have extended the `from_pretrained` function so that `quantization_config` can accept [`WeightOnlyQuantConfig`](https://github.com/intel/intel-extension-for-transformers/blob/main/intel_extension_for_transformers/transformers/utils/quantization_config.py#L28) to implement conversion on the CPU. We not only support PyTorch but also provide LLM Runtime backend based cpp programming language. Here are the example codes.

### LLM Runtime example code
If `use_llm_runtime` is enabled, the LLM Runtime backend is used, the default value is True.
```bash
cd intel_extension_for_transformers/llm/runtime/graph
from intel_extension_for_transformers.transformers import AutoModelForCausalLM, WeightOnlyQuantConfig
model_name_or_path = "Intel/neural-chat-7b-v1-1"
# weight_dtype: int8/int4, compute_dtype: int8/fp32
woq_config = WeightOnlyQuantConfig(weight_dtype="int4", compute_dtype="int8")
model = AutoModelForCausalLM.from_pretrained(
                                            model_name_or_path,
                                            quantization_config=woq_config,
                                            trust_remote_code=True
                                            )
# inference
from transformers import AutoTokenizer, TextStreamer
prompt = "Once upon a time, a little girl"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)
print(outputs)

```
### PyTorch example code
Prepare model name and generate kwargs.
```bash
model_name_or_path = "Intel/neural-chat-7b-v1-1"
generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=4)
prompt = "Once upon a time, a little girl"
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
```
4-bit/8-bit inference with `WeightOnlyQuantConfig` on CPU device.
```bash
from intel_extension_for_transformers.transformers import AutoModelForCausalLM, WeightOnlyQuantConfig
# weight_dtype: int8/int4_fullrange/int4_clip/nf4/fp4_e2m1_bnb/fp4_e2m1/fp8_e5m2/fp8_e4m3
# scale_dtype: fp32/fp8, fp8 only used for weight_dtype "fp8_e5m2", "fp8_e4m3"
woq_config = WeightOnlyQuantConfig(weight_dtype="int4_fullrange", group_size=32)
woq_model = AutoModelForCausalLM.from_pretrained(
                                                    model_name_or_path,
                                                    quantization_config=woq_config,
                                                    use_llm_runtime=False
                                                )
gen_ids = woq_model.generate(input_ids, max_new_tokens=32, **generate_kwargs)
gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
print(gen_text)
```
4-bit/8-bit inference with Huggingface Transformers `BitsAndBytesConfig` on CUDA GPU device.
```bash
from intel_extension_for_transformers.transformers import AutoModelForCausalLM, BitsAndBytesConfig
woq_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
woq_model = AutoModelForCausalLM.from_pretrained(  
                                                    model_name_or_path,
                                                    quantization_config=woq_config,
                                                    use_llm_runtime=False
                                                )
gen_ids = woq_model.generate(input_ids, max_new_tokens=32, **generate_kwargs)
gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
print(gen_text)
```
`load_in_4bit` and `load_in_8bit` both support on CPU and CUDA GPU device. If device set to use GPU, the BitsAndBytesConfig will be used, if the device set to use CPU, the WeightOnlyQuantConfig will be used.
```bash
from intel_extension_for_transformers.transformers import AutoModelForCausalLM
woq_model = AutoModelForCausalLM.from_pretrained(  
                                                    model_name_or_path,
                                                    load_in_4bit=True,
                                                    use_llm_runtime=False
                                                )
gen_ids = woq_model.generate(input_ids, max_new_tokens=32, **generate_kwargs)
gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
print(gen_text)
```
```bash
from intel_extension_for_transformers.transformers import AutoModelForCausalLM
woq_model = AutoModelForCausalLM.from_pretrained(
                                                    model_name_or_path,
                                                    load_in_8bit=True,
                                                    use_llm_runtime=False
                                                )
gen_ids = woq_model.generate(input_ids, max_new_tokens=32, **generate_kwargs)
gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
print(gen_text)
```
