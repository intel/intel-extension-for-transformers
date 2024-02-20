<div align="center">
  
Intel® Extension for Transformers
===========================
<h3>An Innovative Transformer-based Toolkit to Accelerate GenAI/LLM Everywhere</h3>

[![](https://dcbadge.vercel.app/api/server/Wxk3J3ZJkU?compact=true&style=flat-square)](https://discord.gg/Wxk3J3ZJkU)
[![Release Notes](https://img.shields.io/github/v/release/intel/intel-extension-for-transformers)](https://github.com/intel/intel-extension-for-transformers/releases)

[🏭Architecture](./docs/architecture.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[💬NeuralChat](./intel_extension_for_transformers/neural_chat)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[😃Inference on CPU](https://github.com/intel/neural-speed/tree/main)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[😃Inference  on GPU](https://github.com/intel/intel-extension-for-transformers/blob/main/docs/weightonlyquant.md#examples-for-gpu)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[💻Examples](./docs/examples.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[📖Documentations](https://intel.github.io/intel-extension-for-transformers/latest/docs/Welcome.html)
</div>

## 🚀Latest News
* [2024/01] Supported **INT4 inference on Intel GPUs** including Intel Data Center GPU Max Series (e.g., PVC) and Intel Arc A-Series (e.g., ARC). Check out the [examples](https://github.com/intel/intel-extension-for-transformers/blob/main/docs/weightonlyquant.md#examples-for-gpu) and [scripts](https://github.com/intel/intel-extension-for-transformers/blob/main/examples/huggingface/pytorch/text-generation/quantization/run_generation_gpu_woq.py).
* [2024/01] Demonstrated **Intel Hybrid Copilot** in **CES 2024 Great Minds** Session "[Bringing the Limitless Potential of AI Everywhere](https://youtu.be/70J3uO3eLZA?t=1348)".
* [2023/12] Supported **QLoRA on CPUs** to make fine-tuning on client CPU possible. Check out the [blog](https://medium.com/@NeuralCompressor/creating-your-own-llms-on-your-laptop-a08cc4f7c91b) and [readme](https://github.com/intel/intel-extension-for-transformers/blob/main/docs/qloracpu.md) for more details.
* [2023/11] Released **top-1 7B-sized LLM** [**NeuralChat-v3-1**](https://huggingface.co/Intel/neural-chat-7b-v3-1) and [DPO dataset](https://huggingface.co/datasets/Intel/orca_dpo_pairs). Check out the [nice video](https://www.youtube.com/watch?v=bWhZ1u_1rlc) published by [WorldofAI](https://www.youtube.com/@intheworldofai).
* [2023/11] Published a **4-bit chatbot demo** (based on NeuralChat) available on [Intel Hugging Face Space](https://huggingface.co/spaces/Intel/NeuralChat-ICX-INT4). Welcome to have a try! To setup the demo locally, please follow the [instructions](https://github.com/intel/intel-extension-for-transformers/blob/main/intel_extension_for_transformers/neural_chat/docs/notebooks/setup_text_chatbot_service_on_spr.ipynb).

---
<div align="left">

## 🏃Installation
### Quick Install from Pypi
```bash
pip install intel-extension-for-transformers
```
> For system requirements and other installation tips, please refer to [Installation Guide](./docs/installation.md)

## 🌟Introduction
Intel® Extension for Transformers is an innovative toolkit designed to accelerate GenAI/LLM everywhere with the optimal performance of Transformer-based models on various Intel platforms, including Intel Gaudi2, Intel CPU, and Intel GPU. The toolkit provides the below key features and examples:

*  Seamless user experience of model compressions on Transformer-based models by extending [Hugging Face transformers](https://github.com/huggingface/transformers) APIs and leveraging [Intel® Neural Compressor](https://github.com/intel/neural-compressor)

*  Advanced software optimizations and unique compression-aware runtime (released with NeurIPS 2022's paper [Fast Distilbert on CPUs](https://arxiv.org/abs/2211.07715) and [QuaLA-MiniLM: a Quantized Length Adaptive MiniLM](https://arxiv.org/abs/2210.17114), and NeurIPS 2021's paper [Prune Once for All: Sparse Pre-Trained Language Models](https://arxiv.org/abs/2111.05754))

*  Optimized Transformer-based model packages such as [Stable Diffusion](examples/huggingface/pytorch/text-to-image/deployment/stable_diffusion), [GPT-J-6B](examples/huggingface/pytorch/text-generation/deployment), [GPT-NEOX](examples/huggingface/pytorch/language-modeling/quantization#2-validated-model-list), [BLOOM-176B](examples/huggingface/pytorch/language-modeling/inference#BLOOM-176B), [T5](examples/huggingface/pytorch/summarization/quantization#2-validated-model-list), [Flan-T5](examples/huggingface/pytorch/summarization/quantization#2-validated-model-list), and end-to-end workflows such as [SetFit-based text classification](docs/tutorials/pytorch/text-classification/SetFit_model_compression_AGNews.ipynb) and [document level sentiment analysis (DLSA)](workflows/dlsa) 

*  [NeuralChat](intel_extension_for_transformers/neural_chat), a customizable chatbot framework to create your own chatbot within minutes by leveraging a rich set of [plugins](https://github.com/intel/intel-extension-for-transformers/blob/main/intel_extension_for_transformers/neural_chat/docs/advanced_features.md) such as [Knowledge Retrieval](./intel_extension_for_transformers/neural_chat/pipeline/plugins/retrieval/README.md), [Speech Interaction](./intel_extension_for_transformers/neural_chat/pipeline/plugins/audio/README.md), [Query Caching](./intel_extension_for_transformers/neural_chat/pipeline/plugins/caching/README.md), and [Security Guardrail](./intel_extension_for_transformers/neural_chat/pipeline/plugins/security/README.md). This framework supports Intel Gaudi2/CPU/GPU.

*  [Inference](https://github.com/intel/neural-speed/tree/main) of Large Language Model (LLM) in pure C/C++ with weight-only quantization kernels for Intel CPU and Intel GPU (TBD), supporting [GPT-NEOX](https://github.com/intel/neural-speed/tree/main/neural_speed/models/gptneox), [LLAMA](https://github.com/intel/neural-speed/tree/main/neural_speed/models/llama), [MPT](https://github.com/intel/neural-speed/tree/main/neural_speed/models/mpt), [FALCON](https://github.com/intel/neural-speed/tree/main/neural_speed/models/falcon), [BLOOM-7B](https://github.com/intel/neural-speed/tree/main/neural_speed/models/bloom), [OPT](https://github.com/intel/neural-speed/tree/main/neural_speed/models/opt), [ChatGLM2-6B](https://github.com/intel/neural-speed/tree/main/neural_speed/models/chatglm), [GPT-J-6B](https://github.com/intel/neural-speed/tree/main/neural_speed/models/gptj), and [Dolly-v2-3B](https://github.com/intel/neural-speed/tree/main/neural_speed/models/gptneox). Support AMX, VNNI, AVX512F and AVX2 instruction set. We've boosted the performance of Intel CPUs, with a particular focus on the 4th generation Intel Xeon Scalable processor, codenamed [Sapphire Rapids](https://www.intel.com/content/www/us/en/products/docs/processors/xeon-accelerated/4th-gen-xeon-scalable-processors.html).

## 🔓Validated Hardware
<table>
	<tbody>
		<tr>
			<td rowspan="2">Hardware</td>
			<td colspan="2">Fine-Tuning</td>
			<td colspan="2">Inference</td>
		</tr>
		<tr>
			<td>Full</td>
			<td>PEFT</td>
			<td>8-bit</td>
			<td>4-bit</td>
		</tr>
		<tr>
			<td>Intel Gaudi2</td>
			<td>✔</td>
			<td>✔</td>
			<td>WIP (FP8)</td>
			<td>-</td>
		</tr>
		<tr>
			<td>Intel Xeon Scalable Processors</td>
			<td>✔</td>
			<td>✔</td>
			<td>✔ (INT8, FP8)</td>
			<td>✔ (INT4, FP4, NF4)</td>
		</tr>
		<tr>
			<td>Intel Xeon CPU Max Series</td>
			<td>✔</td>
			<td>✔</td>
			<td>✔ (INT8, FP8)</td>
			<td>✔ (INT4, FP4, NF4)</td>
		</tr>
		<tr>
			<td>Intel Data Center GPU Max Series</td>
			<td>WIP </td>
			<td>WIP </td>
			<td>WIP (INT8)</td>
			<td>✔ (INT4)</td>
		</tr>
		<tr>
			<td>Intel Arc A-Series</td>
			<td>-</td>
			<td>-</td>
			<td>WIP (INT8)</td>
			<td>✔ (INT4)</td>
		</tr>
		<tr>
			<td>Intel Core Processors</td>
			<td>-</td>
			<td>✔</td>
			<td>✔ (INT8, FP8)</td>
			<td>✔ (INT4, FP4, NF4)</td>
		</tr>
	</tbody>
</table>


> In the table above, "-" means not applicable or not started yet.

## 🔓Validated Software
<table>
	<tbody>
		<tr>
			<td rowspan="2">Software</td>
			<td colspan="2">Fine-Tuning</td>
			<td colspan="2">Inference</td>
		</tr>
		<tr>
			<td>Full</td>
			<td>PEFT</td>
			<td>8-bit</td>
			<td>4-bit</td>
		</tr>
		<tr>
			<td>PyTorch</td>
			<td>2.0.1+cpu,</br> 2.0.1a0 (gpu)</td>
			<td>2.0.1+cpu,</br> 2.0.1a0 (gpu)</td>
			<td>2.1.0+cpu,</br> 2.0.1a0 (gpu)</td>
			<td>2.1.0+cpu,</br> 2.0.1a0 (gpu)</td>
		</tr>
		<tr>
			<td>Intel® Extension for PyTorch</td>
			<td>2.1.0+cpu,</br> 2.0.110+xpu</td>
			<td>2.1.0+cpu,</br> 2.0.110+xpu</td>
			<td>2.1.0+cpu,</br> 2.0.110+xpu</td>
			<td>2.1.0+cpu,</br> 2.0.110+xpu</td>
		</tr>
		<tr>
			<td>Transformers</td>
			<td>4.35.2(CPU),</br> 4.31.0 (Intel GPU)</td>
			<td>4.35.2(CPU),</br> 4.31.0 (Intel GPU)</td>
			<td>4.35.2(CPU),</br> 4.31.0 (Intel GPU)</td>
			<td>4.35.2(CPU),</br> 4.31.0 (Intel GPU)</td>
		</tr>
		<tr>
			<td>Synapse AI</td>
			<td>1.13.0</td>
			<td>1.13.0</td>
			<td>1.13.0</td>
			<td>1.13.0</td>
		</tr>
		<tr>
			<td>Gaudi2 driver</td>
			<td>1.13.0-ee32e42</td>
			<td>1.13.0-ee32e42</td>
			<td>1.13.0-ee32e42</td>
			<td>1.13.0-ee32e42</td>
		</tr>
                <tr>
                        <td>intel-level-zero-gpu</td>
                        <td>1.3.26918.50-736~22.04 </td>
                        <td>1.3.26918.50-736~22.04 </td>
                        <td>1.3.26918.50-736~22.04 </td>
                        <td>1.3.26918.50-736~22.04 </td>
                </tr>
	</tbody>
</table>

> Please refer to the detailed requirements in [CPU](intel_extension_for_transformers/neural_chat/requirements_cpu.txt), [Gaudi2](intel_extension_for_transformers/neural_chat/requirements_hpu.txt), [Intel GPU](https://github.com/intel/intel-extension-for-transformers/blob/main/requirements-gpu.txt).

## 🔓Validated OS
Ubuntu 20.04/22.04, Centos 8.

## 🌱Getting Started

### Chatbot
Below is the sample code to create your chatbot. See more [examples](intel_extension_for_transformers/neural_chat/docs/full_notebooks.md).

#### Serving (OpenAI-compatible RESTful APIs)
NeuralChat provides OpenAI-compatible RESTful APIs for chat, so you can use NeuralChat as a drop-in replacement for OpenAI APIs.
You can start NeuralChat server either using the Shell command or Python code.

```shell
# Shell Command
neuralchat_server start --config_file ./server/config/neuralchat.yaml
```

```python
# Python Code
from intel_extension_for_transformers.neural_chat import NeuralChatServerExecutor
server_executor = NeuralChatServerExecutor()
server_executor(config_file="./server/config/neuralchat.yaml", log_file="./neuralchat.log")
```

NeuralChat service can be accessible through [OpenAI client library](https://github.com/openai/openai-python), `curl` commands, and `requests` library. See more in [NeuralChat](intel_extension_for_transformers/neural_chat/README.md).

#### Offline

```python
from intel_extension_for_transformers.neural_chat import build_chatbot
chatbot = build_chatbot()
response = chatbot.predict("Tell me about Intel Xeon Scalable Processors.")
```

### Transformers-based extension APIs
Below is the sample code to use the extended Transformers APIs. See more [examples](https://github.com/intel/neural-speed/tree/main).

#### INT4 Inference (CPU)
```python
from transformers import AutoTokenizer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM
model_name = "Intel/neural-chat-7b-v3-1"     
prompt = "Once upon a time, there existed a little girl,"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids

model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True)
outputs = model.generate(inputs)
```

You can also load the low-bit model quantized by GPTQ/AWQ/RTN/AutoRound algorithm.
```python
from transformers import AutoTokenizer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM, WeightOnlyQuantConfig

# Download Hugging Face GPTQ/AWQ model or use local quantize model
model_name = "PATH_TO_MODEL"  # local path to model
woq_config = WeightOnlyQuantConfig(use_gptq=True)   # use_awq=True for AWQ; use_autoround=True for AutoRound
prompt = "Once upon a time, a little girl"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=woq_config, trust_remote_code=True) 
outputs = model.generate(inputs)
```

#### INT4 Inference (GPU)
```python
import intel_extension_for_pytorch as ipex
from intel_extension_for_transformers.transformers.modeling import AutoModelForCausalLM
from transformers import AutoTokenizer

device_map = "xpu"
model_name ="Qwen/Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
prompt = "Once upon a time, there existed a little girl,"
inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device_map)

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,
                                              device_map=device_map, load_in_4bit=True)

model = ipex.optimize_transformers(model, inplace=True, dtype=torch.float16, woq=True, device=device_map)

output = model.generate(inputs)
```
> Note: Please refer to the [example](https://github.com/intel/intel-extension-for-transformers/blob/main/docs/weightonlyquant.md#examples-for-gpu) and [script](https://github.com/intel/intel-extension-for-transformers/blob/main/examples/huggingface/pytorch/text-generation/quantization/run_generation_gpu_woq.py) for more details.

### Langchain-based extension APIs
Below is the sample code to use the extended Langchain APIs. See more [examples](intel_extension_for_transformers/neural_chat/pipeline/plugins/retrieval/README.md).

```python
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_core.vectorstores import VectorStoreRetriever
from intel_extension_for_transformers.langchain.vectorstores import Chroma
retriever = VectorStoreRetriever(vectorstore=Chroma(...))
retrievalQA = RetrievalQA.from_llm(llm=HuggingFacePipeline(...), retriever=retriever)
```

## 🎯Validated  Models
You can access the validated models, accuracy and performance from [Release data](./docs/release_data.md) or [Medium blog](https://medium.com/@NeuralCompressor/llm-performance-of-intel-extension-for-transformers-f7d061556176).

## 📖Documentation
<table>
<thead>
  <tr>
    <th colspan="8" align="center">OVERVIEW</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td colspan="4" align="center"><a href="intel_extension_for_transformers/neural_chat">NeuralChat</a></td>
    <td colspan="4" align="center"><a href="https://github.com/intel/neural-speed/tree/main">Neural Speed</a></td>
  </tr>
  <tr>
    <th colspan="8" align="center">NEURALCHAT</th>
  </tr>
  <tr>
    <td colspan="2" align="center"><a href="intel_extension_for_transformers/neural_chat/docs/notebooks/deploy_chatbot_on_spr.ipynb">Chatbot on Intel CPU</a></td>
    <td colspan="3" align="center"><a href="intel_extension_for_transformers/neural_chat/docs/notebooks/deploy_chatbot_on_xpu.ipynb">Chatbot on Intel GPU</a></td>
    <td colspan="3" align="center"><a href="intel_extension_for_transformers/neural_chat/docs/notebooks/deploy_chatbot_on_habana_gaudi.ipynb">Chatbot on Gaudi</a></td>
  </tr>
  <tr>
    <td colspan="4" align="center"><a href="intel_extension_for_transformers/neural_chat/examples/deployment/talkingbot/pc/build_talkingbot_on_pc.ipynb">Chatbot on Client</a></td>
    <td colspan="4" align="center"><a href="intel_extension_for_transformers/neural_chat/docs/full_notebooks.md">More Notebooks</a></td>
  </tr>
  <tr>
    <th colspan="8" align="center">NEURAL SPEED</th>
  </tr>
 <tr>
    <td colspan="2" align="center"><a href="https://github.com/intel/neural-speed/tree/main/README.md">Neural Speed</a></td>
    <td colspan="2" align="center"><a href="https://github.com/intel/neural-speed/tree/main/README.md#2-neural-speed-straight-forward">Streaming LLM</a></td>
    <td colspan="2" align="center"><a href="https://github.com/intel/neural-speed/tree/main/neural_speed/core#support-matrix">Low Precision Kernels</a></td>
    <td colspan="2" align="center"><a href="https://github.com/intel/neural-speed/tree/main/docs/tensor_parallelism.md">Tensor Parallelism</a></td>
  </tr>
  <tr>
    <th colspan="8" align="center">LLM COMPRESSION</th>
  </tr>
  <tr>
    <td colspan="2" align="center"><a href="docs/smoothquant.md">SmoothQuant (INT8)</a></td>
    <td colspan="3" align="center"><a href="docs/weightonlyquant.md">Weight-only Quantization (INT4/FP4/NF4/INT8)</a></td>
    <td colspan="3" align="center"><a href="docs/qloracpu.md">QLoRA on CPU</a></td>
  </tr>
  <tr>
    <th colspan="8" align="center">GENERAL COMPRESSION</th>
  <tr>
  <tr>
    <td colspan="2" align="center"><a href="docs/quantization.md">Quantization</a></td>
    <td colspan="2" align="center"><a href="docs/pruning.md">Pruning</a></td>
    <td colspan="2" align="center"><a href="docs/distillation.md">Distillation</a></td>
    <td align="center" colspan="2"><a href="examples/huggingface/pytorch/text-classification/orchestrate_optimizations/README.md">Orchestration</a></td>
  </tr>
  <tr>
    <td align="center" colspan="2"><a href="examples/huggingface/pytorch/language-modeling/nas/README.md">Neural Architecture Search</a></td>
    <td align="center" colspan="2"><a href="docs/export.md">Export</a></td>
    <td align="center" colspan="2"><a href="docs/metrics.md">Metrics</a></td>
    <td align="center" colspan="2"><a href="docs/objectives.md">Objectives</a></td>
  </tr>
  <tr>
    <td align="center" colspan="2"><a href="docs/pipeline.md">Pipeline</a></td>
    <td align="center" colspan="2"><a href="examples/huggingface/pytorch/question-answering/dynamic/README.md">Length Adaptive</a></td>
    <td align="center" colspan="2"><a href="docs/examples.md#early-exit">Early Exit</a></td>
    <td align="center" colspan="2"><a href="docs/data_augmentation.md">Data Augmentation</a></td>    
  </tr>
  <tr>
    <th colspan="8" align="center">TUTORIALS & RESULTS</a></th>
  </tr>
  <tr>
    <td colspan="2" align="center"><a href="docs/tutorials/README.md">Tutorials</a></td>
    <td colspan="2" align="center"><a href="https://github.com/intel/neural-speed/blob/main/docs/supported_models.md">LLM List</a></td>
    <td colspan="2" align="center"><a href="docs/examples.md">General Model List</a></td>
    <td colspan="2" align="center"><a href="intel_extension_for_transformers/llm/runtime/deprecated/docs/validated_model.md">Model Performance</a></td>
  </tr>
</tbody>
</table>

## 🙌Demo

* LLM Infinite Inference (up to 4M tokens)

https://github.com/intel/intel-extension-for-transformers/assets/109187816/1698dcda-c9ec-4f44-b159-f4e9d67ab15b

* LLM QLoRA on Client CPU

https://github.com/intel/intel-extension-for-transformers/assets/88082706/9d9bdb7e-65db-47bb-bbed-d23b151e8b31

## 📃Selected Publications/Events
* CES 2024: [CES 2024 Great Minds Keynote: Bringing the Limitless Potential of AI Everywhere: Intel Hybrid Copilot demo](https://youtu.be/70J3uO3eLZA?t=1348) (Jan 2024)
* Blog published on Medium: [Connect an AI agent with your API: Intel Neural-Chat 7b LLM can replace Open AI Function Calling](https://medium.com/11tensors/connect-an-ai-agent-with-your-api-intel-neural-chat-7b-llm-can-replace-open-ai-function-calling-242d771e7c79) (Dec 2023)
* NeurIPS'2023 on Efficient Natural Language and Speech Processing: [Efficient LLM Inference on CPUs](https://arxiv.org/abs/2311.00502) (Nov 2023)
* Blog published on Hugging Face: [Intel Neural-Chat 7b: Fine-Tuning on Gaudi2 for Top LLM Performance](https://huggingface.co/blog/Andyrasika/neural-chat-intel) (Nov 2023)
* Blog published on VMware: [AI without GPUs: A Technical Brief for VMware Private AI with Intel](https://core.vmware.com/resource/ai-without-gpus-technical-brief-vmware-private-ai-intel#section6) (Nov 2023)
  
> View [Full Publication List](./docs/publication.md)

## Additional Content

* [Release Information](./docs/release.md)
* [Contribution Guidelines](./docs/contributions.md)
* [Legal Information](./docs/legal.md)
* [Security Policy](SECURITY.md)
* [Apache License](./LICENSE)


## Acknowledgements
* Excellent open-source projects: [bitsandbytes](https://github.com/TimDettmers/bitsandbytes), [FastChat](https://github.com/lm-sys/FastChat), [fastRAG](https://github.com/IntelLabs/fastRAG), [ggml](https://github.com/ggerganov/ggml), [gptq](https://github.com/IST-DASLab/gptq), [llama.cpp](https://github.com/ggerganov/llama.cpp), [lm-evauation-harness](https://github.com/EleutherAI/lm-evaluation-harness), [peft](https://github.com/huggingface/peft), [trl](https://github.com/huggingface/trl), [streamingllm](https://github.com/mit-han-lab/streaming-llm) and many others.

* Thanks to all the [contributors](./docs/contributors.md).

## 💁Collaborations

Welcome to raise any interesting ideas on model compression techniques and LLM-based chatbot development! Feel free to reach [us](mailto:itrex.maintainers@intel.com), and we look forward to our collaborations on Intel Extension for Transformers!
