Weight Only Quantization (WOQ)
=====

1. [Introduction](#introduction)

2. [Supported Algorithms](#supported-algorithms)

3. [Examples For CPU/CUDA](#examples-for-cpu-and-cuda)

4. [Examples For Intel GPU](#examples-for-intel-gpu)

## Introduction

As large language models (LLMs) become more prevalent, there is a growing need for new and improved quantization methods that can meet the computational demands of these modern architectures while maintaining the accuracy. Compared to [normal quantization](https://github.com/intel/intel-extension-for-transformers/blob/main/docs/quantization.md) like W8A8, weight only quantization is probably a better trade-off to balance the performance and the accuracy, since we will see below that the bottleneck of deploying LLMs is the memory bandwidth and normally weight only quantization could lead to better accuracy.
## Supported Algorithms

| Support Device |  Rtn  |  Awq  |  Teq |  GPTQ  | AutoRound |
|:--------------:|:----------:|:----------:|:----------:|:----:|:----:|
|     Intel CPU        |  &#10004;  |  &#10004;  |  &#10004;  |  &#10004;  |  &#10004;  |
|     Intel GPU        |  &#10004;  |  stay tuned  |  stay tuned  |  &#10004;  |  &#10004;  |

**RTN**[[1\]](https://github.com/intel/intel-extension-for-transformers/blob/548c13ed2e19cde91729530ca26c3b875c1b3d10/docs/weightonlyquant.md#1)(&#9733;&#9733;&#9733;):   Rounding to Nearest (RTN) is an intuitively simple method that rounds values to the nearest integer. It boasts simplicity, requiring no additional datasets, and offers fast quantization. Besides, it could be easily applied in other datatype like NF4(non-uniform). Typically, it performs well on configurations such as W4G32 or W8, but worse than advanced algorithms at lower precision level.


**Teq**[[2\]](https://github.com/intel/intel-extension-for-transformers/blob/548c13ed2e19cde91729530ca26c3b875c1b3d10/docs/weightonlyquant.md#4)(&#9733;&#9733;&#9733;): To our knowledge, it is the first trainable equivalent ransformation method (summited for peer review in 202306). However,  it requires more memory than other methods as model-wise loss is used and the equivalent transformation imposes certain requirements on model architecture.


**GPTQ**[[2\]](https://github.com/intel/intel-extension-for-transformers/blob/548c13ed2e19cde91729530ca26c3b875c1b3d10/docs/weightonlyquant.md#2)(&#9733;&#9733;&#9733;&#9733;): GPTQ is a widely adopted method based on the Optimal Brain Surgeon. It quantizes weight block by block and fine-tunes the remaining unquantized ones to mitigate quantization errors. Occasionally, Non-positive semidefinite matrices may occur, necessitating adjustments to hyperparameters.



**Awq**[[4\]](https://github.com/intel/intel-extension-for-transformers/blob/548c13ed2e19cde91729530ca26c3b875c1b3d10/docs/weightonlyquant.md#3)(&#9733;&#9733;&#9733;&#9733;): AWQ is a popular method that explores weight min-max values and equivalent transformations in a handcrafted space. While effective, the equivalent transformation imposes certain requirements on model architecture, limiting its applicability to broader models or increasing engineering efforts.



**AutoRound**[[5\]](https://github.com/intel/intel-extension-for-transformers/blob/548c13ed2e19cde91729530ca26c3b875c1b3d10/docs/weightonlyquant.md#5)(&#9733;&#9733;&#9733;&#9733;&#9734;): AutoRound utilizes sign gradient descent to optimize rounding values and minmax values of weights within just 200 steps, showcasing impressive performance compared to recent methods like GPTQ/AWQ. Additionally, it offers hypeparameters tuning compatibility to further enhance performance. However, due to its reliance on gradient backpropagation, currently it is not quite fit for backends like ONNX. 

### references
<a id="1">[1]</a> 
Gunho Park, Baeseong Park, Se Jung Kwon, Byeongwook Kim, Youngjoo Lee, and Dongsoo Lee.
nuqmm: Quantized matmul for efficient inference of large-scale generative language models.
arXiv preprint arXiv:2206.09557, 2022.

<a id="2">[2]</a> 
Cheng, W., Cai, Y., Lv, K & Shen, H. (2023).
TEQ: Trainable Equivalent Transformation for Quantization of LLMs. 
arXiv preprint arXiv:2310.10944.

<a id="3">[3]</a> 
Frantar, Elias, et al. "Gptq: Accurate post-training quantization for generative pre-trained transformers." arXiv preprint arXiv:2210.17323 (2022).

<a id="4">[4]</a> 
Lin, Ji, et al.(2023).
AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.
arXiv preprint arXiv:2306.00978.

<a id="5">[5]</a> 
Cheng, W., Zhang, W., Shen, H., Cai, Y., He, X., & Lv, K. (2023).
Optimize weight rounding via signed gradient descent for the quantization of llms. 
arXiv preprint arXiv:2309.05516.

## Examples For CPU AND CUDA

Our motivation is to improve CPU support for weight only quantization, since `bitsandbytes`, `auto-gptq`, `autoawq` only support CUDA GPU device. We have extended the `from_pretrained` function so that `quantization_config` can accept [`RtnConfig`](https://github.com/intel/intel-extension-for-transformers/blob/main/intel_extension_for_transformers/transformers/utils/config.py#L608), [`AwqConfig`](https://github.com/intel/intel-extension-for-transformers/blob/main/intel_extension_for_transformers/transformers/utils/config.py#L793), [`TeqConfig`](https://github.com/intel/intel-extension-for-transformers/blob/main/intel_extension_for_transformers/transformers/utils/config.py#L28), [`GPTQConfig`](https://github.com/intel/intel-extension-for-transformers/blob/main/intel_extension_for_transformers/transformers/utils/config.py#L855), [`AutoroundConfig`](https://github.com/intel/intel-extension-for-transformers/blob/main/intel_extension_for_transformers/transformers/utils/config.py#L912) to implement conversion on the CPU. We not only support PyTorch but also provide LLM Runtime backend based cpp programming language. Here are the example codes.

### Example for CPU device
4-bit/8-bit inference with `RtnConfig`, `AwqConfig`, `TeqConfig`, `GPTQConfig`, `AutoRoundConfig` on CPU device.
```bash
cd examples/huggingface/pytorch/text-generation/quantization
from intel_extension_for_transformers.transformers import AutoModelForCausalLM, RtnConfig
model_name_or_path = "Intel/neural-chat-7b-v3-3"
# weight_dtype: int8/int4, compute_dtype: int8/fp32
woq_config = RtnConfig(bits=4, compute_dtype="int8")
model = AutoModelForCausalLM.from_pretrained(
                                            model_name_or_path,
                                            quantization_config=woq_config,
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
### Example for CUDA GPU device
Prepare model name and generate kwargs.
```bash
model_name_or_path = "Intel/neural-chat-7b-v3-3"
generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=4)
prompt = "Once upon a time, a little girl"
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
```
4-bit/8-bit inference with Huggingface Transformers `BitsAndBytesConfig` on CUDA GPU device.
```bash
from intel_extension_for_transformers.transformers import AutoModelForCausalLM, BitsAndBytesConfig
woq_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
woq_model = AutoModelForCausalLM.from_pretrained(  
                                                    model_name_or_path,
                                                    quantization_config=woq_config,
                                                )
gen_ids = woq_model.generate(input_ids, max_new_tokens=32, **generate_kwargs)
gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
print(gen_text)
```
`load_in_4bit` and `load_in_8bit` both support on CPU and CUDA GPU device. If device set to use GPU, the BitsAndBytesConfig will be used, if the device set to use CPU, the RtnConfig will be used.
```bash
from intel_extension_for_transformers.transformers import AutoModelForCausalLM
woq_model = AutoModelForCausalLM.from_pretrained(  
                                                    model_name_or_path,
                                                    load_in_4bit=True,
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
                                                )
gen_ids = woq_model.generate(input_ids, max_new_tokens=32, **generate_kwargs)
gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
print(gen_text)
```

You can also save and load your quantized low bit model by the below code.

```python
from intel_extension_for_transformers.transformers import AutoModelForCausalLM

model_path = "meta-llama/Llama-2-7b-chat-hf" # your_pytorch_model_path_or_HF_model_name
saved_dir = "4_bit_llama2" # your_saved_model_dir
# quant
model = AutoModelForCausalLM.from_pretrained(model_path, load_in_4bit=True)
# save quant model
model.save_pretrained(saved_dir)
# load quant model
loaded_model = AutoModelForCausalLM.from_pretrained(saved_dir)
```

> Note: For LLM runtime model loading usage, please refer to [neural_speed readme](https://github.com/intel/neural-speed/blob/main/README.md#quick-start-transformer-like-usage)

## Examples For Intel GPU
Intel-extension-for-transformers implement weight-only quantization for intel GPU(PVC/ARC/MTL) with [Intel-extension-for-pytorch](https://github.com/intel/intel-extension-for-pytorch).

Now 4-bit/8-bit inference with `RtnConfig`, `AwqConfig`, `GPTQConfig`, `AutoRoundConfig` are support on intel GPU device.

We support experimental woq inference on intel GPU(PVC/ARC/MTL) with replacing Linear op in PyTorch. Validated models: Qwen-7B, GPT-J-6B (only for PVC/ARC), Llama-7B.  

Here are the example codes.

#### Prepare Dependency Packages
1. Install Oneapi Package  
The Oneapi DPCPP compiler is required to compile intel-extension-for-pytorch. Please follow [the link](https://www.intel.com/content/www/us/en/developer/articles/guide/installation-guide-for-oneapi-toolkits.html) to install the OneAPI to "/opt/intel folder".

2. Build and Install PyTorch and Intel-extension-for-pytorch
```python
python -m pip install torch==2.1.0a0  -f https://developer.intel.com/ipex-whl-stable-xpu

source /opt/intel/oneapi/setvars.sh

# Build IPEX from Source Code
git clone https://github.com/intel/intel-extension-for-pytorch.git ipex-gpu
cd ipex-gpu
git submodule update --init --recursive
export USE_AOT_DEVLIST='pvc,ats-m150'  # Comment this line if you are compiling for MTL
export BUILD_WITH_CPU=OFF

pip install -r requirements.txt

python setup.py install
```

3. Install Intel-extension-for-transformers and Neural-compressor
```pythpon
pip install neural-compressor
pip install intel-extension-for-transformers
```

4. Quantization Model and Inference
```python
import intel_extension_for_pytorch as ipex
from intel_extension_for_transformers.transformers.modeling import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch

device = "xpu"
model_name = "Qwen/Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
prompt = "Once upon a time, there existed a little girl,"
inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

qmodel = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="xpu", trust_remote_code=True)

# optimize the model with ipex, it will improve performance.
qmodel = ipex.optimize_transformers(qmodel, inplace=True, dtype=torch.float16, quantization_config={}, device="xpu")

output = qmodel.generate(inputs)
```

> Note: If your device memory is not enough, please quantize and save the model first, then rerun the example with loading the model as below, If your device memory is enough, skip below instruction, just quantization and inference.

5. Saving and Loading quantized model
 * First step: Quantize and save model
```python

from intel_extension_for_transformers.transformers.modeling import AutoModelForCausalLM

qmodel = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B", load_in_4bit=True, device_map="xpu", trust_remote_code=True)

# Please note, saving model should be executed before ipex.optimize_transformers function is called. 
model.save_pretrained("saved_dir")
```
 * Second step: Load model and inference(In order to reduce memory usage, you may need to end the quantize process and rerun the script to load the model.)
```python
# Load model
loaded_model = AutoModelForCausalLM.from_pretrained("saved_dir", trust_remote_code=True)

# Before executed the loaded model, you can call ipex.optimize_transformers function.
loaded_model = ipex.optimize_transformers(loaded_model, inplace=True, dtype=torch.float16, quantization_config={}, device="xpu")

output = loaded_model.generate(inputs)

```

6. You can directly use [example script](https://github.com/intel/intel-extension-for-transformers/blob/main/examples/huggingface/pytorch/text-generation/quantization/run_generation_gpu_woq.py)
```python
python run_generation_gpu_woq.py --woq --benchmark 
```

>Note:
> * Saving quantized model should be executed before the optimize_transformers function is called.
> * The optimize_transformers function is designed to optimize transformer-based models within frontend Python modules, with a particular focus on Large Language Models (LLMs). It provides optimizations for both model-wise and content-generation-wise. The detail of `optimize_transformers`, please refer to [the link](https://github.com/intel/intel-extension-for-pytorch/blob/xpu-main/docs/tutorials/llm/llm_optimize_transformers.md).


### Example of AutoRound on Intel GPU

For the specific usage of parameters for AutoRoundConfig, please refer to the definition [class AutoRoundConfig](https://github.com/intel/intel-extension-for-transformers/blob/629b9d40caf97c963dc76f908e4cb66cc6f72eeb/intel_extension_for_transformers/transformers/utils/config.py#L930)

```python
import torch
import intel_extension_for_pytorch as ipex
from transformers import AutoConfig, AutoTokenizer
from intel_extension_for_transformers.transformers import (
    AutoModelForCausalLM,
    AutoRoundConfig,
)

device = "xpu"
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
prompt = "Once upon a time, a little girl"
inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

quantization_config = AutoRoundConfig(
    tokenizer=tokenizer,
    bits=4,
    group_size=32,
    max_input_length=2048,
    compute_dtype="fp16",
    scale_dtype="fp16",
    weight_dtype="int4_fullrange",
    calib_iters=2,
    calib_len=32,
    nsamples=2,
    lr=0.0025,
    minmax_lr=0.0025,
)
qmodel = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device,
    quantization_config=quantization_config,
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

# optimize the model with ipex, it will improve performance.
qmodel = ipex.optimize_transformers(
    qmodel, inplace=True, dtype=torch.float16, quantization_config=True, device=device
)
output = qmodel.generate(inputs, max_new_tokens=100, do_sample=True)
print(tokenizer.batch_decode(output, skip_special_tokens=True))
```

### Llama3 on MTL
Currently, we only support running llama3 on MTL and only support the following parameters to quantize and inference:
- quantification method: RTN
- group_size: 32
- batch_size: 1
- num_beams 1

```python
import intel_extension_for_pytorch as ipex
from intel_extension_for_transformers.transformers.modeling import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch

device_map = "xpu"
model_name ="meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
prompt = "Once upon a time, there existed a little girl,"
inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device_map)

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,
                                              device_map=device_map, load_in_4bit=True)

model = ipex.optimize_transformers(model, inplace=True, dtype=torch.float16, quantization_config=True, device=device_map)

output = model.generate(inputs)
```
