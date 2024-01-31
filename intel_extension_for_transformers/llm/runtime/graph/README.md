
> LLM Runtime has been renamed as **Neural Speed** and separated as individual project, see [here](https://github.com/intel/neural-speed/tree/main) for more details.


# Neural Speed

Neural Speed is designed to provide the efficient inference of large language models (LLMs) on Intel platforms through the state-of-the-art (SOTA) model compression techniques. The work is highly inspired from [llama.cpp](https://github.com/ggerganov/llama.cpp), which organizes almost all the core code (e.g., kernels) in a single big file with a large number of pre-defined macros, thus making it not easy for developers to support a new model. Our Neural Speed has the following features:

- Modular design to support new models
- [Highly optimized low precision kernels](core/README.md)
- Utilize AMX, VNNI, AVX512F and AVX2 instruction set
- Support CPU (x86 platforms only) and Intel GPU (WIP)
- Support 4bits and 8bits quantization

> Neural Speed is under active development so APIs are subject to change.

## Supported Hardware
| Hardware | Optimization |
|-------------|:-------------:|
|Intel Xeon Scalable Processors | ✔ |
|Intel Xeon CPU Max Series | ✔ |
|Intel Core Processors | ✔ |
|Intel Arc GPU Series | WIP |
|Intel Data Center GPU Max Series | WIP |
|Intel Gaudi2 | Not yet |

## Supported Models

Neural Speed supports the following models:
[model list](https://github.com/intel/neural-speed/blob/main/docs/supported_models.md)

## How to Use
There are methods for utilizing the Neural Speed:
- [Transformer-based API](#How-to-use-Transformer-based-API)


## How to use: Transformer-based API
### 1. Install
Install from binary, please ensure that your GCC version is higher than 10.
```shell
pip install intel-extension-for-transformers
pip install -r requirements.txt  # under graph folder
```
> Some models only support specific versions of transformers. Please refer to the table above or official documentation.

### 2. Run LLM with Transformer-based API

You can use Python API to run Hugging Face model simply. Here is the sample code:
```python
from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM
model_name = "Intel/neural-chat-7b-v3-1"     # Hugging Face model_id or local model
prompt = "Once upon a time, there existed a little girl,"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)

model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True)
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)
```

To directly load a GPTQ/AWQ/AutoRound model, here is the sample code:
```python
from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM, WeightOnlyQuantConfig

# Download Hugging Face GPTQ model to local path
model_name = "PATH_TO_MODEL"  # local path to model
woq_config = WeightOnlyQuantConfig(use_gptq=True) # use_awq=True for AWQ models, and use_autoround=True for AutoRound models
prompt = "Once upon a time, a little girl"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=woq_config, trust_remote_code=True)
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)
```

To directly load a GGUF model, here is the sample code:
```python
from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM, WeightOnlyQuantConfig

# Specify the GGUF repo on the Hugginface
model_name = "TheBloke/Llama-2-7B-Chat-GGUF"
# Download the the specific gguf model file from the above repo
model_file = "llama-2-7b-chat.Q4_0.gguf"
# make sure you are granted to access this model on the Huggingface.
tokenizer_name = "meta-llama/Llama-2-7b-chat-hf"

prompt = "Once upon a time"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)
model = AutoModelForCausalLM.from_pretrained(model_name, model_file = model_file)
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)
```

To enable [StreamingLLM for infinite inference](./docs/infinite_inference.md), here is the sample code:
```python
from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM, WeightOnlyQuantConfig
model_name = "Intel/neural-chat-7b-v3-1"     # Hugging Face model_id or local model
woq_config = WeightOnlyQuantConfig(compute_dtype="int8", weight_dtype="int4")
prompt = "Once upon a time, there existed a little girl,"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=woq_config)

# Paper: https://arxiv.org/pdf/2309.17453.pdf
# Recommend n_keep=4 to do attention sinks (four initial tokens) and n_discard=-1 to drop half rencetly tokens when meet length threshold
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300, ctx_size=100, n_keep=4, n_discard=-1)
```

To use whisper to Audio-to-text, here is the sample code
```python
from intel_extension_for_transformers.transformers import AutoModelForCausalLM, WeightOnlyQuantConfig
model_name = "Local path for whisper"     # please use local path
woq_config = WeightOnlyQuantConfig(use_ggml=True) #Currently, only Q40 is supported
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=woq_config)
model('Local audio file')
```

https://github.com/intel/intel-extension-for-transformers/assets/109187816/1698dcda-c9ec-4f44-b159-f4e9d67ab15b

Argument description of WeightOnlyQuantConfig ([supported MatMul combinations](#supported-matrix-multiplication-data-types-combinations)):
| Argument          |  Type       | Description                                                                             |
| --------------    | ----------  | -----------------------------------------------------------------------                 |
| compute_dtype     | String      | Data type of Gemm computation: int8/bf16/fp16/fp32 (default: fp32)                           |
| weight_dtype      | String      | Data type of quantized weight: int4/int8/fp8(=fp8_e4m3)/fp8_e5m2/fp4(=fp4_e2m1)/nf4 (default int4)                                 |
| alg               | String      | Quantization algorithm: sym/asym (default sym)                                          |
| group_size        | Int         | Group size: Int, 32/128/-1 (per channel) (default: 32)                                                           |
| scale_dtype       | String      | Data type of scales: fp32/bf16/fp8 (default fp32)                                           |
| use_ggml          | Bool        | Enable ggml for quantization and inference (default: False)                             |
| use_quant         | Bool        | Determine whether or not the model will be quantized. (default: True)                  |

Argument description of generate function:
| Argument          |  Type       | Description                                                                             |
| --------------    | ----------  | -----------------------------------------------------------------------                 |
| inputs            | Lists[Int]  | Input ids after tokenizer                                                               |
| interactive       | Bool        | Interactive mode, use history commands when True (default: False)                       |
| n_keep            | Int         | Number of tokens to keep from the initial prompt (default: 0, -1 = all)                 |
| n_discard         | Int         | Number of tokens will be discarded (default: -1, -1 = half of tokens will be discarded) |
| shift_roped_k     | Bool        | Use ring-buffer and thus do not re-computing after reaching ctx_size (default: False)   |
| ignore_prompt     | Bool        | Generate outputs w/o prompt (default: False)                                            |
| batch_size        | Int         | Batch size for prompt processing (default: 512)                                         |
| ctx_size          | Int         | Size of the prompt context (default: 512)                                               |
| seed              | Int         | NG seed (default: -1, use random seed for < 0)                                          |
| threads           | Int         | Number of threads to use during computation (default: min(available_core_num, OMP_NUM_THREADS))                                       |
| memory_dtype      | str         | Data type of the KV memory; one of f16, f32, auto (enables Fused Attention when possible otherwise fallback to f16) (default: auto)   |
| repetition_penalty| Float       | Please refer to [Transformer's generate](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/text_generation#generation) |
| num_beams         | Int         | Please refer to [Transformer's generate](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/text_generation#generation) |
| do_sample         | Int         | Please refer to [Transformer's generate](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/text_generation#generation) |
| top_k             | Int         | Please refer to [Transformer's generate](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/text_generation#generation) |
| top_p             | Int         | Please refer to [Transformer's generate](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/text_generation#generation) |
| temperature       | Float       | Please refer to [Transformer's generate](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/text_generation#generation) |
| min_new_tokens    | Int         | Please refer to [Transformer's generate](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/text_generation#generation) |
| length_penalty    | Float       | Please refer to [Transformer's generate](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/text_generation#generation) |
| early_stopping    | Bool        | Please refer to [Transformer's generate](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/text_generation#generation) |
| max_new_tokens    | Int         | Please refer to [Transformer's generate](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/text_generation#generation) |
| streamer          | Class       | Please refer to [Transformer's generate](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/text_generation#generation) |
| stopping_criteria | Class       | Please refer to [Transformer's generate](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/text_generation#generation) |
| pad_token         | Int         | pad_token_id of [Transformer's generate](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/text_generation#generation) |

### 3. Multi-Round Chat

Chat with LLaMA2:
```python
from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM, WeightOnlyQuantConfig

# Please change to local path to model, llama2 does not support online conversion, currently.
model_name = "meta-llama/Llama-2-7b-chat-hf"
woq_config = WeightOnlyQuantConfig(compute_dtype="int8", weight_dtype="int4")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
streamer = TextStreamer(tokenizer)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=woq_config, trust_remote_code=True)

while True:
    prompt = input("> ").strip()
    if prompt == "quit":
        break
    b_prompt = "[INST]{}[/INST]".format(prompt)  # prompt template for llama2
    inputs = tokenizer(b_prompt, return_tensors="pt").input_ids
    outputs = model.generate(inputs, streamer=streamer, interactive=True, ignore_prompt=True, do_sample=True)
```

Chat with ChatGLM2:
```python
from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM, WeightOnlyQuantConfig

model_name = "THUDM/chatglm2-6b"  # or local path to model
woq_config = WeightOnlyQuantConfig(compute_dtype="int8", weight_dtype="int4")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
streamer = TextStreamer(tokenizer)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=woq_config, trust_remote_code=True)

while True:
    prompt = input("> ").strip()
    if prompt == "quit":
        break
    prompt = tokenizer.build_prompt(prompt)  # prompt template for chatglm2
    inputs = tokenizer([prompt], return_tensors="pt").input_ids
    outputs = model.generate(inputs, streamer=streamer, interactive=True, ignore_prompt=True, do_sample=True, n_keep=2)
```

Chat with Qwen:
```python
from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM, WeightOnlyQuantConfig

model_name = "Qwen/Qwen-7B-Chat"  # or local path to model
woq_config = WeightOnlyQuantConfig(compute_dtype="int8", weight_dtype="int4")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
streamer = TextStreamer(tokenizer)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=woq_config, trust_remote_code=True)

while True:
    prompt = input("> ").strip()
    if prompt == "quit":
        break
    prompt = "\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(prompt)  # prompt template for qwen
    inputs = tokenizer([prompt], return_tensors="pt").input_ids
    outputs = model.generate(inputs, streamer=streamer, interactive=True, ignore_prompt=True, do_sample=True)
```
