# LLM Runtime 

LLM Runtime is designed to provide the efficient inference of large language models (LLMs) on Intel platforms through the state-of-the-art (SOTA) model compression techniques. The work is highly inspired from [llama.cpp](https://github.com/ggerganov/llama.cpp), which organizes almost all the core code (e.g., kernels) in a single big file with a large number of pre-defined macros, thus making it not easy for developers to support a new model. Our LLM Runtime has the following features:

- Modular design to support new models
- Highly optimized low precision kernels
- Utilize AMX, VNNI and AVX512F instruction set
- Support CPU (x86 platforms only) and initial (Intel) GPU
- Support 4bits and 8bits quantization 

> LLM Runtime is under active development so APIs are subject to change.

## Supported Models

We support the following models:
### Text generation models
| model name | INT8 | INT4|
|---|:---:|:---:|
|[LLaMA2-7B](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), [LLaMA2-13B](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)| ✅ | ✅ | 
|[LLaMA-7B](https://huggingface.co/decapoda-research/llama-7b-hf), [LLaMA-13B](https://huggingface.co/decapoda-research/llama-13b-hf)| ✅ | ✅ | 
|[GPT-J-6B](https://huggingface.co/EleutherAI/gpt-j-6b)| ✅ | ✅ | 
|[GPT-NeoX-20B](https://huggingface.co/EleutherAI/gpt-neox-20b)| ✅ | ✅ | 
|[Dolly-v2-3B](https://huggingface.co/databricks/dolly-v2-3b)| ✅ | ✅ | 
|[MPT-7B](https://huggingface.co/mosaicml/mpt-7b), [MPT-30B](https://huggingface.co/mosaicml/mpt-30b)| ✅ | ✅ | 
|[Falcon-7B](https://huggingface.co/tiiuae/falcon-7b), [Falcon-40B](https://huggingface.co/tiiuae/falcon-40b)| ✅ | ✅ | 
|[BLOOM-7B](https://huggingface.co/bigscience/bloomz-7b1)| ✅ | ✅ |
|[OPT-125m](https://huggingface.co/facebook/opt-125m), [OPT-350m](https://huggingface.co/facebook/opt-350m), [OPT-1.3B](https://huggingface.co/facebook/opt-1.3b), [OPT-13B](https://huggingface.co/facebook/opt-13b)| ✅ | ✅ |  
|[ChatGLM-6B](https://huggingface.co/THUDM/chatglm-6b), [ChatGLM2-6B](https://huggingface.co/THUDM/chatglm2-6b)| ✅ | ✅ |

### Code generation models
| model name | INT8 | INT4|
|---|:---:|:---:|
|[Code-LLaMA-7B](https://huggingface.co/codellama/CodeLlama-7b-hf), [Code-LLaMA-13B](https://huggingface.co/codellama/CodeLlama-13b-hf)| ✅ | ✅ | 
|[StarCoder-1B](https://huggingface.co/bigcode/starcoderbase-1b), [StarCoder-3B](https://huggingface.co/bigcode/starcoderbase-3b), [StarCoder-15.5B](https://huggingface.co/bigcode/starcoder)| ✅ | ✅ | 


## How to use

### 1. Build LLM Runtime
Linux
```shell
mkdir build
cd build
cmake .. -G Ninja
ninja
```
Windows: install VisualStudio 2022(a validated veresion), search 'Developer PowerShell for VS 2022' and open it, then run the following cmds.
```powershell
mkdir build
cd build
cmake ..
cmake --build . -j
```

### 2. Run LLM with Python API

You can use the python api to simplely run HF model.
```python
from intel_extension_for_transformers.transformers import AutoModel, WeightOnlyQuantConfig
model_name = "EleutherAI/gpt-j-6b"     # support model id of HF or local PATH to model
woq_config = WeightOnlyQuantConfig(compute_dtype="int8", weight_dtype="int4")
model = AutoModel.from_pretrained(model_name, quantization_config=woq_config, use_llm_runtime=True)
prompt = "Once upon a time, a little girl"
output = model.generate(prompt, streamer, max_new_tokens=30)
```

### 3. Run LLM with Script
You can use the following script to run, including convertion, quantization and inference.
```
python scripts/run_llm.py model-path --weight_dtype int4 -p "She opened the door and see"
```

LLM one-click running script args explanations:
| arg               | explanation                                                             |
| --------------    | ----------------------------------------------------------------------- |
| model           | directory containing model file or model id                 |
| --weight_dtype  | data type of quantized weight (default: int4)         |
| --alg           | quantization algorithm to use: sym/asym (default: sym)      |
| --block_size    | block size (default: 32)                                    |
| --scale_dtype   | fp32/bf16 type for scales (default: fp32)                   |
| --compute_type  | Gemm computation data type: int8/fp32/ggml (default: ggml)  |
| -p / --prompt     | prompt to start generation with (default: empty)                        |
| -n / --n_predict  | number of tokens to predict (default: -1, -1 = infinity)                |
| -t / --threads    | number of threads to use during computation (default: 56)               |
| -b / --batch_size_truncate | batch size for prompt processing (default: 512)                         |
| -c / --ctx_size   | size of the prompt context (default: 512, can not be larger than specific model's context window length)                                                                                |
| -s / --seed       | NG seed (default: -1, use random seed for < 0)                          |
| --repeat_penalty  | penalize repeat sequence of tokens (default: 1.1, 1.0 = disabled)       |
| --color           | colorise output to distinguish prompt and user input from generations   |
| --keep            | number of tokens to keep from the initial prompt (default: 0, -1 = all) |


## Advanced use

### 1. Convert and Quantize LLM model
LLM Runtime assumes the same model format as [llama.cpp](https://github.com/ggerganov/llama.cpp) and [ggml](https://github.com/ggerganov/ggml). You can also convert the model by following the below steps:

```bash

<<<<<<< HEAD
# convert the model directly use model id in Hugging Face. (recommended)
python scripts/convert.py --outtype f32 --outfile ne-f32.bin EleutherAI/gpt-j-6b

# or you can download fp32 model (e.g., LLAMA2) from Hugging Face at first, then convert the pytorch model to ggml format.
git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
python scripts/convert.py --outtype f32 --outfile ne-f32.bin model_path
=======
# convert the pytorch model to ggml format
python scripts/convert.py --outtype f32 --outfile ne-f32.bin model_path

# or convert the model without downloading it by hand (llama and llama2 are WIP) 
python scripts/convert.py --outtype f32 --outfile EleutherAI/gpt-j-6b
>>>>>>> rename script and add build_dir

# quantize weights of fp32 ggml bin
# model_name: llama, llama2, mpt, falcon, gptj, starcoder, dolly
<<<<<<< HEAD
# to neuarl engine graph optimized q4_j with 128 block_size format (recommended)
python scripts/quantize.py --model_name llama2 --model_file ne-f32.bin --out_file ne-q4_j.bin --weight_dtype int4 --block_size 128 --compute_type int8
<<<<<<< HEAD

# Alternativly you could run ggml q4_0 format like following
python scripts/quantize.py --model_name llama2 --model_file ne-f32.bin --out_file ne-q4_0.bin --weight_dtype int4
# or ues neuarl engine graph optimized q4_j with 32 block_size format
=======
=======
# to neural engine graph optimized q4_j with 128 group_size format (recommended)
python scripts/quantize.py --model_name llama2 --model_file ne-f32.bin --out_file ne-q4_j.bin --weight_dtype int4 --group_size 128 --compute_type int8
>>>>>>> rename block_size to group_size

# to ggml q4_0 format
python scripts/quantize.py --model_name llama2 --model_file ne-f32.bin --out_file ne-q4_0.bin --weight_dtype int4
# to neural engine graph optimized q4_j with 32 group_size format

<<<<<<< HEAD
>>>>>>> rename script and add build_dir
python scripts/quantize.py --model_name llama2 --model_file ne-f32.bin --out_file ne-q4_j.bin --weight_dtype int4 --block_size 32 --compute_type int8
=======
python scripts/quantize.py --model_name llama2 --model_file ne-f32.bin --out_file ne-q4_j.bin --weight_dtype int4 --group_size 32 --compute_type int8
>>>>>>> rename block_size to group_size

```
quantization args explanations:
| arg             | explanation                                                 |
| --------------  | ----------------------------------------------------------- |
| --model_file    | path to the fp32 model                                      |
| --out_file      | path to the quantized model                                 |
| --config        | path to the configuration file (default: )                  |
| --nthread       | number of threads to use (default: 1)                       |
| --weight_dtype  | data type of quantized weight (default: int4)         |
| --alg           | quantization algorithm to use: sym/asym (default: sym)      |
| --group_size    | group size (default: 32)                                    |
| --scale_dtype   | fp32/bf16 type for scales (default: fp32)                   |
| --compute_type  | Gemm computation data type: int8/fp32/ggml (default: ggml)  |
<<<<<<< HEAD

### 2. Inference model with C++ script API
=======
| --build_dir     | build directory of binary executable files                  |
>>>>>>> rename script and add build_dir

We supply LLM running script to run supported models with c++ api conveniently.
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

LLM running script args explanations:
| arg               | explanation                                                             |
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
| --build_dir       | build directory of binary executable files                              |

<<<<<<< HEAD
### 3. Tensor Parallelism cross nodes/sockets
=======
### 4. One-click Script 

You can use the following script to run, including convertion, quantization and inference.
```
python scripts/one_click_run.py model-path --weight_dtype int4 -p "She opened the door and see"
```

LLM one-click running script args explanations:
| arg               | explanation                                                             |
| --------------    | ----------------------------------------------------------------------- |
| model             | directory containing model file or model id                             |
| --weight_dtype    | data type of quantized weight (default: int4)                           |
| --alg             | quantization algorithm to use: sym/asym (default: sym)                  |
| --group_size      | group size (default: 32)                                                |
| --scale_dtype     | fp32/bf16 type for scales (default: fp32)                               |
| --compute_type    | Gemm computation data type: int8/fp32/ggml (default: ggml)              |
| -p / --prompt     | prompt to start generation with (default: empty)                        |
| -n / --n_predict  | number of tokens to predict (default: -1, -1 = infinity)                |
| -t / --threads    | number of threads to use during computation (default: 56)               |
| -b / --batch_size_truncate | batch size for prompt processing (default: 512)                         |
| -c / --ctx_size   | size of the prompt context (default: 512, can not be larger than specific model's context window length)                                                                                       |
| -s / --seed       | NG seed (default: -1, use random seed for < 0)                          |
| --repeat_penalty  | penalize repeat sequence of tokens (default: 1.1, 1.0 = disabled)       |
| --color           | colorise output to distinguish prompt and user input from generations   |
| --keep            | number of tokens to keep from the initial prompt (default: 0, -1 = all) |
| --build_dir       | build directory of binary executable files                              |

### 5. Tensor Parallelism cross nodes/sockets
>>>>>>> rename script and add build_dir

We support tensor parallelism strategy for distributed inference/training on multi-node and multi-socket.  You can refer to [tensor_parallelism.md](./tensor_parallelism.md) to enable this feature.

