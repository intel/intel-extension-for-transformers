<div align="center">
  
Intel® Extension for Transformers
===========================
<h3>An Innovative Transformer-based Toolkit to Accelerate GenAI/LLM Everywhere</h3>

[![](https://dcbadge.vercel.app/api/server/Wxk3J3ZJkU?compact=true&style=flat-square)](https://discord.gg/Wxk3J3ZJkU)
[![Release Notes](https://img.shields.io/github/v/release/intel/intel-extension-for-transformers)](https://github.com/intel/intel-extension-for-transformers/releases)

[🏭Architecture](./docs/architecture.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[💬NeuralChat](./intel_extension_for_transformers/neural_chat)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[😃Inference](./intel_extension_for_transformers/llm/runtime/graph)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[💻Examples](./docs/examples.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[📖Documentations](https://intel.github.io/intel-extension-for-transformers/latest/docs/Welcome.html)
</div>

## 🚀Latest News
* <b>NeuralChat has been showcased in [Intel Innovation’23 Keynote](https://www.youtube.com/watch?v=RbKRELWP9y8&t=2954s) and [Google Cloud Next'23](https://cloud.google.com/blog/topics/google-cloud-next/welcome-to-google-cloud-next-23) to demonstrate GenAI/LLM capabilities on Intel Xeon Scalable Processors.</b>
* <b>NeuralChat supports custom chatbot development and deployment on broad Intel HWs such as Xeon Scalable Processors, Gaudi2, Xeon CPU Max Series, Data Center GPU Max Series, Arc Series, and Core Processors. Check out [Notebooks](./intel_extension_for_transformers/neural_chat/docs/full_notebooks.md). </b>
* <b>LLM runtime extends Hugging Face Transformers API to provide seamless low precision inference for popular LLMs, supporting mainstream low precision data types such as INT8/FP8/INT4/FP4/NF4.</b>


```python
# pip install intel-extension-for-transformers
from intel_extension_for_transformers.neural_chat import build_chatbot
chatbot = build_chatbot()
response = chatbot.predict("Tell me about Intel Xeon Scalable Processors.")
```

---
<div align="left">

## 🏃Installation
### Quick Install from Pypi
```bash
pip install intel-extension-for-transformers
```
> For more installation method, please refer to [Installation Page](docs/installation.md)

## 🌟Introduction
Intel® Extension for Transformers is an innovative toolkit to accelerate Transformer-based models on Intel platforms, in particular effective on 4th Intel Xeon Scalable processor Sapphire Rapids (codenamed [Sapphire Rapids](https://www.intel.com/content/www/us/en/products/docs/processors/xeon-accelerated/4th-gen-xeon-scalable-processors.html)). The toolkit provides the below key features and examples:


*  Seamless user experience of model compressions on Transformer-based models by extending [Hugging Face transformers](https://github.com/huggingface/transformers) APIs and leveraging [Intel® Neural Compressor](https://github.com/intel/neural-compressor)


*  Advanced software optimizations and unique compression-aware runtime (released with NeurIPS 2022's paper [Fast Distilbert on CPUs](https://arxiv.org/abs/2211.07715) and [QuaLA-MiniLM: a Quantized Length Adaptive MiniLM](https://arxiv.org/abs/2210.17114), and NeurIPS 2021's paper [Prune Once for All: Sparse Pre-Trained Language Models](https://arxiv.org/abs/2111.05754))


*  Optimized Transformer-based model packages such as [Stable Diffusion](examples/huggingface/pytorch/text-to-image/deployment/stable_diffusion), [GPT-J-6B](examples/huggingface/pytorch/text-generation/deployment), [GPT-NEOX](examples/huggingface/pytorch/language-modeling/quantization#2-validated-model-list), [BLOOM-176B](examples/huggingface/pytorch/language-modeling/inference#BLOOM-176B), [T5](examples/huggingface/pytorch/summarization/quantization#2-validated-model-list), [Flan-T5](examples/huggingface/pytorch/summarization/quantization#2-validated-model-list) and end-to-end workflows such as [SetFit-based text classification](docs/tutorials/pytorch/text-classification/SetFit_model_compression_AGNews.ipynb) and [document level sentiment analysis (DLSA)](workflows/dlsa) 

*  [NeuralChat](intel_extension_for_transformers/neural_chat), a customizable chatbot framework to create your own chatbot within minutes by leveraging a rich set of plugins [Knowledge Retrieval](./intel_extension_for_transformers/neural_chat/pipeline/plugins/retrieval/README.md), [Speech Interaction](./intel_extension_for_transformers/neural_chat/pipeline/plugins/audio/README.md), [Query Caching](./intel_extension_for_transformers/neural_chat/pipeline/plugins/caching/README.md), [Security Guardrail](./intel_extension_for_transformers/neural_chat/pipeline/plugins/security/README.md).


*  [Inference](intel_extension_for_transformers/llm/runtime/graph) of Large Language Model (LLM) in pure C/C++ with weight-only quantization kernels, supporting [GPT-NEOX](intel_extension_for_transformers/llm/runtime/graph/models/gptneox), [LLAMA](intel_extension_for_transformers/llm/runtime/graph/models/llama), [MPT](intel_extension_for_transformers/llm/runtime/graph/models/mpt), [FALCON](intel_extension_for_transformers/llm/runtime/graph/models/falcon), [BLOOM-7B](intel_extension_for_transformers/llm/runtime/graph/models/bloom), [OPT](intel_extension_for_transformers/llm/runtime/graph/models/opt), [ChatGLM2-6B](intel_extension_for_transformers/llm/runtime/graph/models/chatglm), [GPT-J-6B](intel_extension_for_transformers/llm/runtime/graph/models/gptj) and [Dolly-v2-3B](intel_extension_for_transformers/llm/runtime/graph/models/gptneox)


## 🌱Getting Started
### LLM Weight-Only Inference
LLM Runtime weight-only [examples](intel_extension_for_transformers/llm/runtime/graph) provide int4/int8 inference.

#### LLM Runtime int4 Inference 
```python
from intel_extension_for_transformers.transformers import AutoModel, WeightOnlyQuantConfig
prompt = "Once upon a time, a little girl"
config = WeightOnlyQuantConfig(compute_dtype="int8")
model = AutoModel.from_pretrained("mosaicml/mpt-7b", quantization_config=config, use_llm_runtime=True)
print(model.generate(prompt, max_new_tokens=30))
```

#### LLM Runtime int8 Inference
```python
from intel_extension_for_transformers.transformers import AutoModel, WeightOnlyQuantConfig
prompt = "Once upon a time, a little girl"
config = WeightOnlyQuantConfig(compute_dtype="bf16", weight_dtype="int8")
model = AutoModel.from_pretrained("mosaicml/mpt-7b", quantization_config=config, use_llm_runtime=True)
print(model.generate(prompt, max_new_tokens=30))
```

## 🎯Validated PModels
Below, you will find the average values for Lambada (OpenAI), HellaSwag, Winogrande, PIQA, and WikiText.

| Model |  FP32         | INT4 group size 32 | INT4 group size 128 | 
|---------------------|:----------------------:|-----------------------|-----------------------------------|
| [EleutherAI/gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6B) | 0.643 | 0.644 | 0.64 |
| [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) | 0.69 | 0.69 | 0.685 |
| [decapoda-research/llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf) | 0.689 | 0.682 | 0.68 |
| [EleutherAI/gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b) | 0.674 | 0.672 | 0.669 |
| [mosaicml/mpt-7b-chat](https://huggingface.co/mosaicml/mpt-7b-chat) | 0.672 | 0.67 | 0.666 |
| [tiiuae/falcon-7b](https://huggingface.co/tiiuae/falcon-7b) | 0.698 | 0.694 | 0.693 |
| [baichuan-inc/baichuan-7B](https://huggingface.co/baichuan-inc/baichuan-7B) | 0.474 | 0.471 | 0.47 |
| [facebook/opt-6.7b](https://huggingface.co/facebook/opt-6.7b) | 0.65 | 0.647 | 0.643 |
| [databricks/dolly-v2-3b](https://huggingface.co/databricks/dolly-v2-3b) | 0.613 | 0.609 | 0.609 |
| [databricks/dolly-v2-7b](https://huggingface.co/databricks/dolly-v2-7b) | 0.631 | 0.627 | 0.623 |
| [tiiuae/falcon-40b-instruct](https://huggingface.co/tiiuae/falcon-40b-instruct) | 0.756 | 0.757 | 0.755 |


## 📖Documentation
<table>
<thead>
  <tr>
    <th colspan="8" align="center">OVERVIEW</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td colspan="2" align="center"><a href="docs">Model Compression</a></td>
    <td colspan="2" align="center"><a href="intel_extension_for_transformers/neural_chat">NeuralChat</a></td>
    <td colspan="2" align="center"><a href="intel_extension_for_transformers/llm/runtime/deprecated/docs">Neural Engine</a></td>
    <td colspan="2" align="center"><a href="intel_extension_for_transformers/llm/runtime/deprecated/kernels/README.md">Kernel Libraries</a></td>
  </tr>
  <tr>
    <th colspan="8" align="center">MODEL COMPRESSION</th>
  </tr>
  <tr>
    <td colspan="2" align="center"><a href="docs/quantization.md">Quantization</a></td>
    <td colspan="2" align="center"><a href="docs/pruning.md">Pruning</a></td>
    <td colspan="2" align="center" colspan="2"><a href="docs/distillation.md">Distillation</a></td>
    <td align="center" colspan="2"><a href="examples/huggingface/pytorch/text-classification/orchestrate_optimizations/README.md">Orchestration</a></td>
  </tr>
  <tr>
    <td align="center" colspan="2"><a href="examples/huggingface/pytorch/language-modeling/nas/README.md">Neural Architecture Search</a></td>
    <td align="center" colspan="2"><a href="docs/export.md">Export</a></td>
    <td align="center" colspan="2"><a href="docs/metrics.md">Metrics</a>/<a href="docs/objectives.md">Objectives</a></td>
    <td align="center" colspan="2"><a href="docs/pipeline.md">Pipeline</a></td>
  </tr>
  <tr>
    <th colspan="8" align="center">NEURAL ENGINE</th>
  </tr>
  <tr>
    <td colspan="2" align="center"><a href="intel_extension_for_transformers/llm/runtime/deprecated/docs/onnx_compile.md">Model Compilation</a></td>
    <td colspan="2" align="center"><a href="intel_extension_for_transformers/llm/runtime/deprecated/docs/add_customized_pattern.md">Custom Pattern</a></td>
    <td colspan="2" align="center"><a href="intel_extension_for_transformers/llm/runtime/deprecated/docs/deploy_and_integration.md">Deployment</a></td>
    <td colspan="2" align="center"><a href="intel_extension_for_transformers/llm/runtime/deprecated/docs/engine_profiling.md">Profiling</a></td>
  </tr>
  <tr>
    <th colspan="8" align="center">KERNEL LIBRARIES</th>
  </tr>
    <td colspan="2" align="center"><a href="intel_extension_for_transformers/llm/runtime/deprecated/kernels/docs/kernel_desc">Sparse GEMM Kernels</a></td>
    <td colspan="2" align="center"><a href="intel_extension_for_transformers/llm/runtime/deprecated/kernels/docs/kernel_desc">Custom INT8 Kernels</a></td>
    <td colspan="2" align="center"><a href="intel_extension_for_transformers/llm/runtime/deprecated/kernels/docs/profiling.md">Profiling</a></td>
    <td colspan="2" align="center"><a href="intel_extension_for_transformers/llm/runtime/deprecated/test/kernels/benchmark/benchmark.md">Benchmark</a></td>
  <tr>
    <th colspan="8" align="center">ALGORITHMS</th>
  </tr>
  <tr>
    <td align="center" colspan="4"><a href="examples/huggingface/pytorch/question-answering/dynamic/README.md">Length Adaptive</a></td>
    <td align="center" colspan="4"><a href="docs/data_augmentation.md">Data Augmentation</a></td>    
  </tr>
  <tr>
    <th colspan="8" align="center">TUTORIALS AND RESULTS</a></th>
  </tr>
  <tr>
    <td colspan="2" align="center"><a href="docs/tutorials/pytorch">Tutorials</a></td>
    <td colspan="2" align="center"><a href="docs/examples.md">Supported Models</a></td>
    <td colspan="2" align="center"><a href="intel_extension_for_transformers/llm/runtime/deprecated/docs/validated_model.md">Model Performance</a></td>
    <td colspan="2" align="center"><a href="intel_extension_for_transformers/llm/runtime/deprecated/kernels/docs/validated_data.md">Kernel Performance</a></td>
  </tr>
</tbody>
</table>


## 📃Selected Publications/Events
* Keynote: [Intel Innovation 2023 Livestream - Day2](https://www.youtube.com/watch?v=RbKRELWP9y8&t=2954s) (Sep 2023)
* Blog published on Medium: [NeuralChat: A Customizable Chatbot Framework](https://medium.com/intel-analytics-software/make-your-own-chatbot-within-a-few-minutes-with-neuralchat-a-customizable-chatbot-framework-139b4bdec8d1) (Sep 2023)
* Blog published on Medium: [Faster Stable Diffusion Inference with Intel Extension for Transformers](https://medium.com/intel-analytics-software/faster-stable-diffusion-inference-with-intel-extension-for-transformers-on-intel-platforms-7e0f563186b0) (July 2023)
* Blog of Intel Developer News: [The Moat Is Trust, Or Maybe Just Responsible AI](https://www.intel.com/content/www/us/en/developer/articles/technical/moat-is-trust-minimizing-risks-generative-ai.html) (July 2023)
* Blog of Intel Developer News: [Create Your Own Custom Chatbot](https://www.intel.com/content/www/us/en/developer/articles/technical/train-large-language-models-create-custom-chatbot.html) (July 2023)
* Blog of Intel Developer News: [Accelerate Llama 2 with Intel AI Hardware and Software Optimizations](https://www.intel.com/content/www/us/en/developer/articles/news/llama2.html) (July 2023)
* Arxiv: [An Efficient Sparse Inference Software Accelerator for Transformer-based Language Models on CPUs](https://arxiv.org/abs/2306.16601) (June 2023)
* Blog published on Medium: [Simplify Your Custom Chatbot Deployment](https://medium.com/intel-analytics-software/simplify-your-custom-chatbot-deployment-on-intel-platforms-c8a911d906cf) (June 2023)


> View [Full Publication List](./docs/publication.md).
## Additional Content

* [Release Information](./docs/release.md)
* [Contribution Guidelines](./docs/contributions.md)
* [Legal Information](./docs/legal.md)
* [Security Policy](SECURITY.md)


## Acknowledgements
* Excellent open-source projects: [bitsandbytes](https://github.com/TimDettmers/bitsandbytes), [FastChat](https://github.com/lm-sys/FastChat), [fastRAG](https://github.com/IntelLabs/fastRAG), [ggml](https://github.com/ggerganov/ggml), [gptq](https://github.com/IST-DASLab/gptq), [llama.cpp](https://github.com/ggerganov/llama.cpp), [lm-evauation-harness](https://github.com/EleutherAI/lm-evaluation-harness), [peft](https://github.com/huggingface/peft), [trl](https://github.com/huggingface/trl), and many others.


## 💁Collaborations

Welcome to raise any interesting ideas on model compression techniques and LLM-based chatbot development! Feel free to reach [us](mailto:itrex.maintainers@intel.com) and look forward to our collaborations on Intel Extension for Transformers!
