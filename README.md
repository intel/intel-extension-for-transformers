<div align="center">
  
Intel® Extension for Transformers
===========================
<h3>An Innovative Transformer-based Toolkit to Accelerate GenAI/LLM Everywhere</h3>

[![](https://dcbadge.vercel.app/api/server/Wxk3J3ZJkU?compact=true&style=flat-square)](https://discord.gg/Wxk3J3ZJkU)
[![Release Notes](https://img.shields.io/github/v/release/intel/intel-extension-for-transformers)](https://github.com/intel/intel-extension-for-transformers/releases)

[🏭Architecture](./docs/architecture.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[💬NeuralChat](./intel_extension_for_transformers/neural_chat)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[😃Inference](./intel_extension_for_transformers/llm/runtime/graph)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[💻Examples](./docs/examples.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[📖Documentations](https://intel.github.io/intel-extension-for-transformers/latest/docs/Welcome.html)
</div>

## 🚀Latest News
* [2023/12] Supported **QLoRA on CPUs** to make fine-tuning on client CPU possible. Check out the [blog](https://medium.com/@NeuralCompressor/creating-your-own-llms-on-your-laptop-a08cc4f7c91b) for more details.
* [2023/11] Demonstrated up to **3x LLM inference speedup** using **[Assisted Generation](https://huggingface.co/blog/assisted-generation)** (also called Speculative Decoding) from Hugging Face with Intel optimizations! Check out [more details](https://github.com/intel/intel-extension-for-transformers/blob/main/examples/huggingface/pytorch/text-generation/assisted_generation/README.md).
* [2023/11] Supported **LLM QLoRA on CPU**, first time enabling LLM fine-tuning on client CPUs (see [more details](https://github.com/intel/intel-extension-for-transformers/blob/main/docs/qloracpu.md)).
* [2023/11] Refreshed **top-1 7B-sized LLM** by releasing [**NeuralChat-v3-1**](https://huggingface.co/Intel/neural-chat-7b-v3-1). Check out the [nice video](https://www.youtube.com/watch?v=bWhZ1u_1rlc) published by [WorldofAI](https://www.youtube.com/@intheworldofai).
* [2023/11] Released [**NeuralChat-v3**](https://huggingface.co/Intel/neural-chat-7b-v3), new **top-1 7B-sized LLM** available on Hugging Face. The model is fine-tuned on Intel Gaudi2 with supervised fine-tuning and direct preference optimization. Check out the [blog](https://medium.com/@NeuralCompressor/the-practice-of-supervised-finetuning-and-direct-preference-optimization-on-habana-gaudi2-a1197d8a3cd3).
* [2023/11] Published a **4-bit chatbot demo** (based on NeuralChat) available on [Intel Hugging Face Space](https://huggingface.co/spaces/Intel/NeuralChat-ICX-INT4). Welcome to have a try! To setup the demo locally, please follow the [instructions](https://github.com/intel/intel-extension-for-transformers/blob/main/intel_extension_for_transformers/neural_chat/docs/notebooks/setup_text_chatbot_service_on_spr.ipynb).
* [2023/11] Released [**Fast, accurate, and infinite LLM inference**](https://github.com/intel/intel-extension-for-transformers/blob/main/intel_extension_for_transformers/llm/runtime/graph/docs/infinite_inference.md) with improved [StreamingLLM](https://arxiv.org/abs/2309.17453) on Intel CPUs!
* [2023/11] Our paper [Efficient LLM Inference on CPUs](https://arxiv.org/abs/2311.00502) has been accepted by **NeurIPS'23** on Efficient Natural Language and Speech Processing. Thanks to all the collaborators!
* [2023/10] LLM runtime, an Intel-optimized [GGML](https://github.com/ggerganov/ggml) compatible runtime, demonstrates **up to 15x performance gain in 1st token generation and 1.5x in other token generation** over the default [llama.cpp](https://github.com/ggerganov/llama.cpp).
* [2023/10] LLM runtime now supports LLM inference with **infinite-length inputs up to 4 million tokens**, inspired from [StreamingLLM](https://arxiv.org/abs/2309.17453).
* [2023/09] NeuralChat has been showcased in [**Intel Innovation’23 Keynote**](https://www.youtube.com/watch?v=RbKRELWP9y8&t=2954s) and [Google Cloud Next'23](https://cloud.google.com/blog/topics/google-cloud-next/welcome-to-google-cloud-next-23) to demonstrate GenAI/LLM capabilities on Intel Xeon Scalable Processors.
* [2023/08] NeuralChat supports **custom chatbot development and deployment within minutes** on broad Intel HWs such as Xeon Scalable Processors, Gaudi2, Xeon CPU Max Series, Data Center GPU Max Series, Arc Series, and Core Processors. Check out [Notebooks](./intel_extension_for_transformers/neural_chat/docs/full_notebooks.md).
* [2023/07] LLM runtime extends Hugging Face Transformers API to provide seamless low precision inference for popular LLMs, supporting low precision data types such as INT3/INT4/FP4/NF4/INT5/INT8/FP8.

---
<div align="left">

## 🏃Installation
### Quick Install from Pypi
```bash
pip install intel-extension-for-transformers
```
> For more installation methods, please refer to [Installation Page](./docs/installation.md)

## 🌟Introduction
Intel® Extension for Transformers is an innovative toolkit designed to accelerate GenAI/LLM everywhere with the optimal performance of Transformer-based models on various Intel platforms, including Intel Gaudi2, Intel CPU, and Intel GPU (TBD). The toolkit provides the below key features and examples:

*  Seamless user experience of model compressions on Transformer-based models by extending [Hugging Face transformers](https://github.com/huggingface/transformers) APIs and leveraging [Intel® Neural Compressor](https://github.com/intel/neural-compressor)

*  Advanced software optimizations and unique compression-aware runtime (released with NeurIPS 2022's paper [Fast Distilbert on CPUs](https://arxiv.org/abs/2211.07715) and [QuaLA-MiniLM: a Quantized Length Adaptive MiniLM](https://arxiv.org/abs/2210.17114), and NeurIPS 2021's paper [Prune Once for All: Sparse Pre-Trained Language Models](https://arxiv.org/abs/2111.05754))

*  Optimized Transformer-based model packages such as [Stable Diffusion](examples/huggingface/pytorch/text-to-image/deployment/stable_diffusion), [GPT-J-6B](examples/huggingface/pytorch/text-generation/deployment), [GPT-NEOX](examples/huggingface/pytorch/language-modeling/quantization#2-validated-model-list), [BLOOM-176B](examples/huggingface/pytorch/language-modeling/inference#BLOOM-176B), [T5](examples/huggingface/pytorch/summarization/quantization#2-validated-model-list), [Flan-T5](examples/huggingface/pytorch/summarization/quantization#2-validated-model-list), and end-to-end workflows such as [SetFit-based text classification](docs/tutorials/pytorch/text-classification/SetFit_model_compression_AGNews.ipynb) and [document level sentiment analysis (DLSA)](workflows/dlsa) 

*  [NeuralChat](intel_extension_for_transformers/neural_chat), a customizable chatbot framework to create your own chatbot within minutes by leveraging a rich set of plugins [Knowledge Retrieval](./intel_extension_for_transformers/neural_chat/pipeline/plugins/retrieval/README.md), [Speech Interaction](./intel_extension_for_transformers/neural_chat/pipeline/plugins/audio/README.md), [Query Caching](./intel_extension_for_transformers/neural_chat/pipeline/plugins/caching/README.md), and [Security Guardrail](./intel_extension_for_transformers/neural_chat/pipeline/plugins/security/README.md). This framework supports Intel Gaudi2/CPU/GPU.

*  [Inference](intel_extension_for_transformers/llm/runtime/graph) of Large Language Model (LLM) in pure C/C++ with weight-only quantization kernels for Intel CPU and Intel GPU (TBD), supporting [GPT-NEOX](intel_extension_for_transformers/llm/runtime/graph/models/gptneox), [LLAMA](intel_extension_for_transformers/llm/runtime/graph/models/llama), [MPT](intel_extension_for_transformers/llm/runtime/graph/models/mpt), [FALCON](intel_extension_for_transformers/llm/runtime/graph/models/falcon), [BLOOM-7B](intel_extension_for_transformers/llm/runtime/graph/models/bloom), [OPT](intel_extension_for_transformers/llm/runtime/graph/models/opt), [ChatGLM2-6B](intel_extension_for_transformers/llm/runtime/graph/models/chatglm), [GPT-J-6B](intel_extension_for_transformers/llm/runtime/graph/models/gptj), and [Dolly-v2-3B](intel_extension_for_transformers/llm/runtime/graph/models/gptneox). Support AMX, VNNI, AVX512F and AVX2 instruction set. We've boosted the performance of Intel CPUs, with a particular focus on the 4th generation Intel Xeon Scalable processor, codenamed [Sapphire Rapids](https://www.intel.com/content/www/us/en/products/docs/processors/xeon-accelerated/4th-gen-xeon-scalable-processors.html).

## 🔓Supported Hardware
[Validated software configurations](intel_extension_for_transformers/docs/release_data.md)
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
			<td>✔ (INT8)</td>
			<td>✔ (INT4)</td>
		</tr>
		<tr>
			<td>Intel Xeon CPU Max Series</td>
			<td>✔</td>
			<td>✔</td>
			<td>✔ (INT8)</td>
			<td>✔ (INT4)</td>
		</tr>
		<tr>
			<td>Intel Core Processors</td>
			<td>n/a</td>
			<td>✔</td>
			<td>✔ (INT8)</td>
			<td>✔ (INT4)</td>
		</tr>
	</tbody>
</table>
> **Notes**: - means not applicable or not started

Intel Gaudi2 details: Driver version 1.13.0-ee32e42, synpase AI v1.13.0

## 🌱Getting Started
You have the choice to initiate your journey with the Chatbot using either Intel Gaudi2 or an Intel CPU.
Below is the sample code to enable the chatbot. See more [examples](intel_extension_for_transformers/neural_chat/docs/full_notebooks.md).

### Chatbot 
```python
# pip install intel-extension-for-transformers
from intel_extension_for_transformers.neural_chat import build_chatbot
chatbot = build_chatbot()
response = chatbot.predict("Tell me about Intel Xeon Scalable Processors.")
```

Below is the sample code to enable weight-only INT4/INT8 inference. See more [examples](intel_extension_for_transformers/llm/runtime/graph).

### INT4 Inference (CPU only)
```python
from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM
model_name = "Intel/neural-chat-7b-v1-1"     # Hugging Face model_id or local model
prompt = "Once upon a time, there existed a little girl,"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)

model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True)
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)
```

### INT8 Inference (CPU only)
```python
from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM
model_name = "Intel/neural-chat-7b-v1-1"     # Hugging Face model_id or local model
prompt = "Once upon a time, there existed a little girl,"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)

model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)
```

## 🎯Validated  Models
You can access the latest int4 performance and accuracy at [int4 blog](https://medium.com/@NeuralCompressor/llm-performance-of-intel-extension-for-transformers-f7d061556176).

Additionally, we are preparing to introduce Baichuan, Mistral, and other models into [LLM Runtime (Intel Optimized llamacpp)](./intel_extension_for_transformers/llm/runtime/graph). For comprehensive accuracy and performance data, though not the most up-to-date, please refer to the [Release data](./docs/release_data.md). 

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
    <td colspan="4" align="center"><a href="intel_extension_for_transformers/llm/runtime/graph">LLM Runtime</a></td>
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
    <th colspan="8" align="center">LLM RUNTIME</th>
  </tr>
 <tr>
    <td colspan="2" align="center"><a href="intel_extension_for_transformers/llm/runtime/graph/README.md">LLM Runtime</a></td>
    <td colspan="2" align="center"><a href="intel_extension_for_transformers/llm/runtime/graph/README.md#2-run-llm-with-python-api">Streaming LLM</a></td>
    <td colspan="2" align="center"><a href="intel_extension_for_transformers/llm/runtime/graph/core/README.md">Low Precision Kernels</a></td>
    <td colspan="2" align="center"><a href="intel_extension_for_transformers/llm/runtime/graph/docs/tensor_parallelism.md">Tensor Parallelism</a></td>
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
    <td colspan="2" align="center"><a href="intel_extension_for_transformers/llm/runtime/graph#supported-models">LLM List</a></td>
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
* NeurIPS'2023 on Efficient Natural Language and Speech Processing: [Efficient LLM Inference on CPUs](https://arxiv.org/abs/2311.00502) (Nov 2023)
* NeurIPS'2023 on Diffusion Models: [Effective Quantization for Diffusion Models on CPUs](https://arxiv.org/pdf/2311.16133.pdf) (Nov 2023)
* Blog published on datalearner: [Analysis of the top ten popular open source LLM of HuggingFace in the fourth week of November 2023 - the explosion of multi-modal large models and small-scale models](https://www.datalearner.com/blog/1051701014024122) (Nov 2023)
* Blog published on zaker: [With this toolkit, the inference performance of large models can be accelerated by 40 times](https://app.myzaker.com/news/article.php?pk=656857148e9f0961d70ac3d3) (Nov 2023)
* Blog published on geeky-gadgets: [New Intel Neural-Chat 7B LLM tops Hugging Face leaderboard beating original Mistral 7B] (https://www.geeky-gadgets.com/intel-neural-chat-7b-llm/) (Nov 2023)
* Blog published on Huggingface: [Intel Neural-Chat 7b: Fine-Tuning on Gaudi2 for Top LLM Performance](https://huggingface.co/blog/Andyrasika/neural-chat-intel) (Nov 2023)
* Video on YouTube: [Neural Chat 7B v3-1 Installation on Windows - Step by Step](https://www.youtube.com/watch?v=7_urstS-noU) (Nov 2023)
* Video on YouTube: [Intel's Neural-Chat 7b: Most Powerful 7B Model! Beats GPT-4!?](https://www.youtube.com/watch?v=bWhZ1u_1rlc) (Nov 2023)
* Blog published on marktechpost: [Intel Researchers Propose a New Artificial Intelligence Approach to Deploy LLMs on CPUs More Efficiently](https://www.marktechpost.com/2023/11/09/intel-researchers-propose-a-new-artificial-intelligence-approach-to-deploy-llms-on-cpus-more-efficiently) (Nov 2023)
* Blog published on VMware: [AI without GPUs: A Technical Brief for VMware Private AI with Intel](https://core.vmware.com/resource/ai-without-gpus-technical-brief-vmware-private-ai-intel#section6) (Nov 2023)
* News releases on VMware: [VMware Collaborates with Intel to Unlock Private AI Everywhere](https://news.vmware.com/releases/vmware-explore-2023-barcelona-intel-private-ai) (Nov 2023)
* Video on YouTube: [Build Your Own ChatBot with Neural Chat | Intel Software](https://www.youtube.com/watch?v=KWT6yKfu4n0) (Oct 2023)
* Blog published on Medium: [Layer-wise Low-bit Weight Only Quantization on a Laptop](https://medium.com/@NeuralCompressor/high-performance-low-bit-layer-wise-weight-only-quantization-on-a-laptop-712580899396) (Oct 2023)
* Blog published on Medium: [Intel-Optimized Llama.CPP in Intel Extension for Transformers](https://medium.com/@NeuralCompressor/llm-performance-of-intel-extension-for-transformers-f7d061556176) (Oct 2023)
* Blog published on Medium: [Reduce the Carbon Footprint of Large Language Models](https://medium.com/intel-analytics-software/reduce-large-language-model-carbon-footprint-with-intel-neural-compressor-and-intel-extension-for-dfadec3af76a) (Oct 2023)




> View [Full Publication List](./docs/publication.md).
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
