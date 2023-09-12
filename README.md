<div align="center">
  
Intel® Extension for Transformers
===========================
<h3> An innovative toolkit to accelerate Transformer-based models on Intel platforms</h3>

[![](https://dcbadge.vercel.app/api/server/mnZDVAbm?compact=true&style=flat-square)](https://discord.gg/mnZDVAbm)
[![Release Notes](https://img.shields.io/github/v/release/intel/intel-extension-for-transformers)](https://github.com/intel/intel-extension-for-transformers/releases)

[🏭Architecture](./docs/architecture.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[💬NeuralChat](./intel_extension_for_transformers/neural_chat)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[😃Inference](./intel_extension_for_transformers/llm/runtime/graph)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[💻Examples](./docs/examples.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[📖Documentations](https://intel.github.io/intel-extension-for-transformers/latest/docs/Welcome.html)
</div>

## 🚀Latest News
* <b>NeuralChat, a customizable chatbot framework under Intel® Extension for Transformers, is now available for you to create your own chatbot within minutes!</b> It supports a rich set of plugins [Knowledge Retrieval](./intel_extension_for_transformers/neural_chat/pipeline/plugins/retrieval/README.md), [Speech Interaction](./intel_extension_for_transformers/neural_chat/pipeline/plugins/audio/README.md), [Query Caching](./intel_extension_for_transformers/neural_chat/pipeline/plugins/caching/README.md), [Security Guardrail](./intel_extension_for_transformers/neural_chat/pipeline/plugins/security/README.md), and multiple architectures such as Intel® Xeon® Scalable Processors and Habana Gaudi® Accelerator.</b> Check out the below sample code and have a try now!

```python
# pip install intel-extension-for-transformers
from intel_extension_for_transformers.neural_chat import build_chatbot
chatbot = build_chatbot()
response = chatbot.predict("Tell me about Intel Xeon Scalable Processors.")
```

* <b>[💬NeuralChat v1.1](https://huggingface.co/Intel/neural-chat-7b-v1-1), a fine-tuned chat model based on [MPT-7B](https://huggingface.co/mosaicml/mpt-7b) using a mixed set of instruction datasets, is available on Hugging Face, together with the release of INT8 quantization recipes and benchmark results.</b>

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

*  [NeuralChat](intel_extension_for_transformers/neural_chat), a customizable chatbot framework to create your own chatbot within minutes by leveraging a rich set of plugins and SOTA optimizations


*  [Inference](intel_extension_for_transformers/llm/runtime/graph) of Large Language Model (LLM) in pure C/C++ with weight-only quantization kernels. It already enabled [GPT-NEOX](intel_extension_for_transformers/llm/runtime/graph/models/gptneox), [LLAMA](intel_extension_for_transformers/llm/runtime/graph/models/llama), [MPT](intel_extension_for_transformers/llm/runtime/graph/models/mpt), [FALCON](intel_extension_for_transformers/llm/runtime/graph/models/falcon), [BLOOM-7B](intel_extension_for_transformers/llm/runtime/graph/models/bloom), [OPT](intel_extension_for_transformers/llm/runtime/graph/models/opt), [ChatGLM2-6B](intel_extension_for_transformers/llm/runtime/graph/models/chatglm), [GPT-J-6B](intel_extension_for_transformers/llm/runtime/graph/models/gptj) and [Dolly-v2-3B](intel_extension_for_transformers/llm/runtime/graph/models/gptneox)


## 🌱Getting Started
### Sentiment Analysis with Quantization
#### Prepare Dataset
```python
from datasets import load_dataset, load_metric
from transformers import AutoConfig,AutoModelForSequenceClassification,AutoTokenizer

raw_datasets = load_dataset("glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
raw_datasets = raw_datasets.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=128), batched=True)
```
#### Quantization
```python
from intel_extension_for_transformers.transformers import QuantizationConfig, metrics, objectives
from intel_extension_for_transformers.transformers.trainer import NLPTrainer

config = AutoConfig.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english",num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english",config=config)
model.config.label2id = {0: 0, 1: 1}
model.config.id2label = {0: 'NEGATIVE', 1: 'POSITIVE'}
# Replace transformers.Trainer with NLPTrainer
# trainer = transformers.Trainer(...)
trainer = NLPTrainer(model=model, 
    train_dataset=raw_datasets["train"], 
    eval_dataset=raw_datasets["validation"],
    tokenizer=tokenizer
)
q_config = QuantizationConfig(metrics=[metrics.Metric(name="eval_loss", greater_is_better=False)])
model = trainer.quantize(quant_config=q_config)

input = tokenizer("I like Intel Extension for Transformers", return_tensors="pt")
output = model(**input).logits.argmax().item()
```

> For more quick samples, please refer to [Get Started Page](docs/get_started.md). For more validated examples, please refer to [Support Model Matrix](docs/examples.md)

## 🎯Validated Performance


| Model |  FP32 | BF16 | INT8 |
|---------------------|:----------------------:|-----------------------|-----------------------------------|
| [EleutherAI/gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6B) | 4163.67 (ms) | 1879.61 (ms) | 1612.24 (ms) |
| [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) | 10.33 (s) | 3.02 (s) | N/A |

> Note*: GPT-J-6B software/hardware configuration please refer to [text-generation](./examples/huggingface/pytorch/text-generation/README.md). Stable-diffusion software/hardware configuration please refer to [text-to-image](./examples/huggingface/pytorch/text-to-image/deployment/stable_diffusion/README.md)



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
* Blog published on Medium: [Faster Stable Diffusion Inference with Intel Extension for Transformers](https://medium.com/intel-analytics-software/faster-stable-diffusion-inference-with-intel-extension-for-transformers-on-intel-platforms-7e0f563186b0) (July 2023)
* Blog of Intel Developer News: [The Moat Is Trust, Or Maybe Just Responsible AI](https://www.intel.com/content/www/us/en/developer/articles/technical/moat-is-trust-minimizing-risks-generative-ai.html) (July 2023)
* Blog of Intel Developer News: [Create Your Own Custom Chatbot](https://www.intel.com/content/www/us/en/developer/articles/technical/train-large-language-models-create-custom-chatbot.html) (July 2023)
* Blog of Intel Developer News: [Accelerate Llama 2 with Intel AI Hardware and Software Optimizations](https://www.intel.com/content/www/us/en/developer/articles/news/llama2.html) (July 2023)
* Arxiv: [An Efficient Sparse Inference Software Accelerator for Transformer-based Language Models on CPUs](https://arxiv.org/abs/2306.16601) (June 2023)
* Blog published on Medium: [Simplify Your Custom Chatbot Deployment](https://medium.com/intel-analytics-software/simplify-your-custom-chatbot-deployment-on-intel-platforms-c8a911d906cf) (June 2023)
* Blog published on Medium: [Create Your Own Custom Chatbot](https://medium.com/intel-analytics-software/create-your-own-chatbot-on-cpus-b8d186cfefb2) (April 2023)

> View [Full Publication List](./docs/publication.md).
## Additional Content

* [Release Information](./docs/release.md)
* [Contribution Guidelines](./docs/contributions.md)
* [Legal Information](./docs/legal.md)
* [Security Policy](SECURITY.md)

## 💁Collaborations

Welcome to raise any interesting ideas on model compression techniques and LLM-based chatbot development! Feel free to reach [us](mailto:inc.maintainers@intel.com) and look forward to our collaborations on Intel Extension for Transformers!
