# ITREX Graph 

ITREX Graph is an experimental c++ bare metal LLM inference solution that mainly references and borrows from the [llama.cpp](https://github.com/ggerganov/llama.cpp) project. llama.cpp puts almost all core code and kernels in a single file and use a large number of macros, making it difficult for developers to read and modify. The graph has the following features:

- Simple and hierarchical structure, you can add your own high-performance implementation.
- Applied quantization into high 4 bitsfrom int8, higher performance and accuracy compared with the original one.
- Utilized AMX, VNNI and AVX512F instruction set, more instructions support on the way.
- Currently only supports x86 platforms, and initial Intel GPU support.

In short, ITREX Graph is an experimental feature and may keep changing.

### Supported Models

Now we supports following models.
| model name | INT8 | INT4|
|---|:---:|:---:|
|[GPT-J-6B](https://huggingface.co/EleutherAI/gpt-j-6b)| ✅ | ✅ | 
|[GPT-NeoX-20B](https://huggingface.co/EleutherAI/gpt-neox-20b)| ✅ | ✅ | 
|[Dolly-v2-3B](https://huggingface.co/databricks/dolly-v2-3b)| ✅ | ✅ | 
|[LLaMA-7B](https://huggingface.co/decapoda-research/llama-7b-hf)| ✅ | ✅ | 
|[LLaMA-13B](https://huggingface.co/decapoda-research/llama-13b-hf)| ✅ | ✅ | 
|[LLaMA2-7B](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)| ✅ | ✅ | 
|[LLaMA2-13B](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)| ✅ | ✅ | 
|[Code-LLaMA-7B](https://huggingface.co/codellama/CodeLlama-7b-hf)| ✅ | ✅ | 
|[Code-LLaMA-13B](https://huggingface.co/codellama/CodeLlama-13b-hf)| ✅ | ✅ | 
|[MPT-7B](https://huggingface.co/mosaicml/mpt-7b)| ✅ | ✅ | 
|[MPT-30B](https://huggingface.co/mosaicml/mpt-30b)| ✅ | ✅ | 
|[Falcon-7B](https://huggingface.co/tiiuae/falcon-7b)| ✅ | ✅ | 
|[Falcon-40B](https://huggingface.co/tiiuae/falcon-40b)| ✅ | ✅ | 
|[StarCoder-1B](https://huggingface.co/bigcode/starcoderbase-1b)| ✅ | ✅ | 
|[StarCoder-3B](https://huggingface.co/bigcode/starcoderbase-3b)| ✅ | ✅ | 
|[StarCoder-15.5B](https://huggingface.co/bigcode/starcoder)| ✅ | ✅ | 


## How to use

### 1. Build Graph
```shell
mkdir build
cd build
cmake .. -G Ninja
ninja
```

### 2. Convert Models
Currently, Graph uses the same model format as [llama.cpp](https://github.com/ggerganov/llama.cpp) and [ggml](https://github.com/ggerganov/ggml). You can also convert the model yourself.

You can use `git clone` command like (for example, gpt-j-6b): `git clone https://huggingface.co/EleutherAI/gpt-j-6b`.

get fp32 model from HuggingFcae links in Supported Models and put it in `graph` folder.
Convert process had two steps: 1. get fp32 model with llama.cpp format 2. quantize the fp32 model into model with low precision (int8, int4, etc.) We recommend you to use int4 model for better LLM inference latency.

```bash
# convert the pytorch model to ggml format
python scripts/convert_model.py --outtype f32 --outfile ne-f32.bin model_path/model_id

# quantize weights of float32 ggml bin
# to ggml q4_0 format
python scripts/quant_bin.py --model_name llama --model_file ne-f32.bin --out_file ne-q4_0.bin --bits 4
# to neuarl engine graph optimized q4_j with 32 block_size format
python scripts/quant_bin.py --model_name llama --model_file ne-f32.bin --out_file ne-q4_j.bin --bits 4 --block_size 32 --compute_type int8
# to neuarl engine graph optimized q4_j with 128 block_size format (recommend)
python scripts/quant_bin.py --model_name llama --model_file ne-f32.bin --out_file ne-q4_j.bin --bits 4 --block_size 128 --compute_type int8
```
quantization args explanations:
| arg             | explanation                                                 |
| --------------  | ----------------------------------------------------------- |
| --model_file    | path to the fp32 model                                      |
| --out_file      | path to the quantized model                                 |
| --config        | path to the configuration file (default: )                  |
| --nthread       | number of threads to use (default: 1)                       |
| --bits          | number of bits to use for quantization (default: 4)         |
| --alg           | quantization algorithm to use: sym/asym (default: sym)      |
| --block_size    | block size (default: 32)                                    |
| --scale_dtype   | fp32/bf16 type for scales (default: fp32)                   |
| --compute_type  | Gemm computation data type: int8/fp32/ggml (default: ggml)  |

Running GPT-NEOX / MPT / FALCON / / GPT-J / STARCODER model, please use `chat_gptneox` / `chat_mpt` / `chat_falcon` / `chat_starcoder` (Please type **prompt about codes** when use `STARCODER`. For example, `-p "def fibonnaci("`).

### 3. Run Models

We supply LLM chat python script to run supported models conveniently.

```bash
# recommed to use numactl to bind cores in Intel cpus for better performance
# if you use different core numbers, please also  change -t arg value
# please type prompt about codes when run `StarCoder`, for example, -p "def fibonnaci(".
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python scripts/chat_llm.py --model_name llama -m ne-q4_j.bin -c 512 -b 1024 -n 256 -t 56 --color -p "She opened the door and see"

# if you want to generate fixed outputs, please set --seed arg, for example:
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python scripts/chat_llm.py --model_name llama -m ne-q4_j.bin -c 512 -b 1024 -n 256 -t 56 --color -p "She opened the door and see" --seed 12

# if you want to reduce repeated generated texts, please set --repeat_penalty (value > 1.0, default = 1.0), for example:
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python scripts/chat_llm.py --model_name llama -m ne-q4_j.bin -c 512 -b 1024 -n 256 -t 56 --color -p "She opened the door and see" --repeat_penalty 1.2
```

Chat script args explanations:
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

