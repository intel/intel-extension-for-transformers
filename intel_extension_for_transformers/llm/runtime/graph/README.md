# LLM Runtime 

LLM Runtime is designed to provide the efficient inference of large language models (LLMs) on Intel platforms through the state-of-the-art (SOTA) model compression techniques. The work is highly inspired from [llama.cpp](https://github.com/ggerganov/llama.cpp), which organizes almost all the core code (e.g., kernels) in a single big file with a large number of pre-defined macros, thus making it not easy for developers to support a new model. Our LLM Runtime has the following features:

- Modular design to support new models
- [Highly optimized low precision kernels](core/README.md)
- Utilize AMX, VNNI and AVX512F instruction set
- Support CPU (x86 platforms only) and initial (Intel) GPU
- Support 4bits and 8bits quantization 

> LLM Runtime is under active development so APIs are subject to change.

## Supported Models

LLM Runtime supports the following models:
### Text Generation
| model name | INT8 | INT4|
|---|:---:|:---:|
|[LLaMA2-7B](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), [LLaMA2-13B](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf), [LLaMA2-70B](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)| ✅ | ✅ | 
|[LLaMA-7B](https://huggingface.co/decapoda-research/llama-7b-hf), [LLaMA-13B](https://huggingface.co/decapoda-research/llama-13b-hf)| ✅ | ✅ | 
|[GPT-J-6B](https://huggingface.co/EleutherAI/gpt-j-6b)| ✅ | ✅ | 
|[GPT-NeoX-20B](https://huggingface.co/EleutherAI/gpt-neox-20b)| ✅ | ✅ | 
|[Dolly-v2-3B](https://huggingface.co/databricks/dolly-v2-3b)| ✅ | ✅ | 
|[MPT-7B](https://huggingface.co/mosaicml/mpt-7b), [MPT-30B](https://huggingface.co/mosaicml/mpt-30b)| ✅ | ✅ | 
|[Falcon-7B](https://huggingface.co/tiiuae/falcon-7b), [Falcon-40B](https://huggingface.co/tiiuae/falcon-40b)| ✅ | ✅ | 
|[BLOOM-7B](https://huggingface.co/bigscience/bloomz-7b1)| ✅ | ✅ |
|[OPT-125m](https://huggingface.co/facebook/opt-125m), [OPT-350m](https://huggingface.co/facebook/opt-350m), [OPT-1.3B](https://huggingface.co/facebook/opt-1.3b), [OPT-13B](https://huggingface.co/facebook/opt-13b)| ✅ | ✅ |  
|[ChatGLM-6B](https://huggingface.co/THUDM/chatglm-6b), [ChatGLM2-6B](https://huggingface.co/THUDM/chatglm2-6b)| ✅ | ✅ |

### Code Generation
| model name | INT8 | INT4|
|---|:---:|:---:|
|[Code-LLaMA-7B](https://huggingface.co/codellama/CodeLlama-7b-hf), [Code-LLaMA-13B](https://huggingface.co/codellama/CodeLlama-13b-hf)| ✅ | ✅ | 
|[StarCoder-1B](https://huggingface.co/bigcode/starcoderbase-1b), [StarCoder-3B](https://huggingface.co/bigcode/starcoderbase-3b), [StarCoder-15.5B](https://huggingface.co/bigcode/starcoder)| ✅ | ✅ | 


## How to Use

### 1. Install LLM Runtime
Install from binary
```shell
pip install intel-extension-for-transformers
```

Build from source
```shell
# Linux
git submodule update --init --recursive
mkdir build
cd build
cmake .. -G Ninja
ninja
```

```powershell
# Windows
# Install VisualStudio 2022 and open 'Developer PowerShell for VS 2022'
mkdir build
cd build
cmake ..
cmake --build . -j
```

### 2. Run LLM with Python API

You can use Python API to run Hugging Face model simply. Here is the sample code:
```python
from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModel, WeightOnlyQuantConfig
model_name = "Intel/neural-chat-7b-v1-1"     # Hugging Face model_id or local model
woq_config = WeightOnlyQuantConfig(compute_dtype="int8", weight_dtype="int4")
prompt = "Once upon a time, a little girl"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)

model = AutoModel.from_pretrained(model_name, quantization_config=woq_config, trust_remote_code=True)
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)

```

### 3. Run LLM with Python Script
You can run LLM with one-click python script including convertion, quantization and inference.
```
python scripts/run.py model-path --weight_dtype int4 -p "She opened the door and see"
```

Augument description of run.py:
| Augument         | Description                                                              |
| --------------    | ----------------------------------------------------------------------- |
| model           | directory containing model file or model id                               |
| --weight_dtype  | data type of quantized weight (default: int4)                             |
| --alg           | quantization algorithm to use: sym/asym (default: sym)                    |
| --group_size    | group size (default: 32)                                                  |
| --scale_dtype   | fp32/bf16 type for scales (default: fp32)                                 |
| --compute_dtype | data type of Gemm computation: int8/bf16/fp32 (default: int8)             |
| --use_ggml      | enable ggml for quantization and inference                                |
| -p / --prompt     | prompt to start generation with (default: empty)                        |
| -n / --n_predict  | number of tokens to predict (default: -1, -1 = infinity)                |
| -t / --threads    | number of threads to use during computation (default: 56)               |
| -b / --batch_size_truncate | batch size for prompt processing (default: 512)                |
| -c / --ctx_size   | size of the prompt context (default: 512, can not be larger than specific model's context window length)                                                                                       |
| -s / --seed       | NG seed (default: -1, use random seed for < 0)                          |
| --repeat_penalty  | penalize repeat sequence of tokens (default: 1.1, 1.0 = disabled)       |
| --color           | colorise output to distinguish prompt and user input from generations   |
| --keep            | number of tokens to keep from the initial prompt (default: 0, -1 = all) |


## Advanced Usage
Besides the one-click script, LLM Runtime also offers the detailed script: 1) convert and quantize, and 2) inference.

### 1. Convert and Quantize LLM
LLM Runtime assumes the compatible model format as [llama.cpp](https://github.com/ggerganov/llama.cpp) and [ggml](https://github.com/ggerganov/ggml). You can also convert the model by following the below steps:

```bash

# convert the model directly use model id in Hugging Face. (recommended)
python scripts/convert.py --outtype f32 --outfile ne-f32.bin EleutherAI/gpt-j-6b

# or you can download fp32 model (e.g., LLAMA2) from Hugging Face at first, then convert the pytorch model to ggml format.
git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
python scripts/convert.py --outtype f32 --outfile ne-f32.bin model_path

# quantize weights of fp32 ggml bin
# model_name: llama, llama2, mpt, falcon, gptj, starcoder, dolly
# optimized INT4 model with group size 128 (recommended)
python scripts/quantize.py --model_name llama2 --model_file ne-f32.bin --out_file ne-q4_j.bin --weight_dtype int4 --group_size 128 --compute_dtype int8

# Alternativly you could run ggml q4_0 format like following
python scripts/quantize.py --model_name llama2 --model_file ne-f32.bin --out_file ne-q4_0.bin --weight_dtype int4 --use_ggml
# optimized INT4 model with group size 32
python scripts/quantize.py --model_name llama2 --model_file ne-f32.bin --out_file ne-q4_j.bin --weight_dtype int4 --group_size 32 --compute_dtype int8

```
Augument description of quantize.py:
| Augument        | Description                                                 |
| --------------  | ----------------------------------------------------------- |
| --model_file    | path to the fp32 model                                      |
| --out_file      | path to the quantized model                                 |
| --config        | path to the configuration file (default: "")                |
| --nthread       | number of threads to use (default: 1)                       |
| --weight_dtype  | data type of quantized weight: int4/int8 (default: int4)    |
| --alg           | quantization algorithm to use: sym/asym (default: sym)      |
| --group_size    | group size (default: 32)                                    |
| --scale_dtype   | data type of scales: bf16/fp32 (default: fp32)              |
| --compute_dtype | data type of Gemm computation: int8/bf16/fp32 (default: int8)  |
| --use_ggml      | enable ggml for quantization and inference                  |


### 2. Inference LLM

We provide LLM inference script to run the quantized model. Please reach [us](mailto:itrex.maintainers@intel.com) if you want to run using C++ API directly.
```bash
# recommed to use numactl to bind cores in Intel cpus for better performance
# if you use different core numbers, please also  change -t arg value
# please type prompt about codes when run `StarCoder`, for example, -p "def fibonnaci(".
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python scripts/inference.py --model_name llama -m ne-q4_j.bin -c 512 -b 1024 -n 256 -t 56 --color -p "She opened the door and see"

# if you want to generate fixed outputs, please set --seed arg, for example:
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python scripts/inference.py --model_name llama -m ne-q4_j.bin -c 512 -b 1024 -n 256 -t 56 --color -p "She opened the door and see" --seed 12

# if you want to reduce repeated generated texts, please set --repeat_penalty (value > 1.0, default = 1.0), for example:
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python scripts/inference.py --model_name llama -m ne-q4_j.bin -c 512 -b 1024 -n 256 -t 56 --color -p "She opened the door and see" --repeat_penalty 1.2
```

Augument description of inference.py:
| Augument          | Description                                                             |
| --------------    | ----------------------------------------------------------------------- |
| --model_name      | model name                                                              |
| -m / --model      | path to the executed model                                              |
| -p / --prompt     | prompt to start generation with (default: empty)                        |
| -n / --n_predict  | number of tokens to predict (default: -1, -1 = infinity)                |
| -t / --threads    | number of threads to use during computation (default: 56)               |
| -b / --batch_size | batch size for prompt processing (default: 512)                         |
| -c / --ctx_size   | size of the prompt context (default: 512, can not be larger than specific model's context window length)                                                                                |
| -s / --seed       | NG seed (default: -1, use random seed for < 0)                          |
| --repeat_penalty  | penalize repeat sequence of tokens (default: 1.1, 1.0 = disabled)       |
| --color           | colorise output to distinguish prompt and user input from generations   |
| --keep            | number of tokens to keep from the initial prompt (default: 0, -1 = all) |
| --glm_tokenizer   | the path of the chatglm tokenizer (default: THUDM/chatglm-6b)           |


### 3. Tensor Parallelism cross nodes/sockets

We support tensor parallelism strategy for distributed inference/training on multi-node and multi-socket.  You can refer to [tensor_parallelism.md](./tensor_parallelism.md) to enable this feature.
