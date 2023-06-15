<div align="center">
  
Intel® Extension for Transformers
===========================
<h3> An innovative toolkit to accelerate Transformer-based models on Intel platforms</h3>

[Architecture](./docs/architecture.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[NeuralChat](./workflows/chatbot)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Examples](./docs/examples.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Documentations](https://intel.github.io/intel-extension-for-transformers/latest/docs/Welcome.html)
</div>

---
<div align="left">

Intel® Extension for Transformers is an innovative toolkit to accelerate Transformer-based models on Intel platforms, in particular effective on 4th Intel Xeon Scalable processor Sapphire Rapids (codenamed [Sapphire Rapids](https://www.intel.com/content/www/us/en/products/docs/processors/xeon-accelerated/4th-gen-xeon-scalable-processors.html)). The toolkit provides the below key features and examples:


*  Seamless user experience of model compressions on Transformer-based models by extending [Hugging Face transformers](https://github.com/huggingface/transformers) APIs and leveraging [Intel® Neural Compressor](https://github.com/intel/neural-compressor)


*  Advanced software optimizations and unique compression-aware runtime (released with NeurIPS 2022's paper [Fast Distilbert on CPUs](https://arxiv.org/abs/2211.07715) and [QuaLA-MiniLM: a Quantized Length Adaptive MiniLM](https://arxiv.org/abs/2210.17114), and NeurIPS 2021's paper [Prune Once for All: Sparse Pre-Trained Language Models](https://arxiv.org/abs/2111.05754))


*  Optimized Transformer-based model packages such as [Stable Diffusion](examples/huggingface/pytorch/text-to-image/deployment/stable_diffusion), [GPT-J-6B](examples/huggingface/pytorch/text-generation/deployment), [GPT-NEOX](examples/huggingface/pytorch/language-modeling/quantization#2-validated-model-list), [BLOOM-176B](examples/huggingface/pytorch/language-modeling/inference#BLOOM-176B), [T5](examples/huggingface/pytorch/summarization/quantization#2-validated-model-list), [Flan-T5](examples/huggingface/pytorch/summarization/quantization#2-validated-model-list) and end-to-end workflows such as [SetFit-based text classification](docs/tutorials/pytorch/text-classification/SetFit_model_compression_AGNews.ipynb) and [document level sentiment analysis (DLSA)](workflows/dlsa) 

*  [NeuralChat](workflows/chatbot), a custom Chatbot trained on Intel CPUs through parameter-efficient fine-tuning [PEFT](https://github.com/huggingface/peft) on domain knowledge


## Installation
### Install from Pypi
```bash
pip install intel-extension-for-transformers
```
> For more installation method, please refer to [Installation Page](docs/installation.md)

## Getting Started
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
from intel_extension_for_transformers.optimization import QuantizationConfig, metrics, objectives
from intel_extension_for_transformers.optimization.trainer import NLPTrainer

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

## Validated Performance


| Model |  FP32 | BF16 | INT8 |
|---------------------|:----------------------:|-----------------------|-----------------------------------|
| [EleutherAI/gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6B) | 4163.67 (ms) | 1879.61 (ms) | 1612.24 (ms) |
| [decapoda-research/llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf) | 4356.79 (ms) | 1941.63 (ms) | 1773.22 (ms) |
| [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) | 10.33 (s) | 3.02 (s) | N/A |

> Note*: 
>
> (1) Settings for [EleutherAI/gpt-j-6B and decapoda-research/llama-7b-hf](./examples/huggingface/pytorch/text-generation/deployment/) : 
>
> Input_Length = 32, Output_Length = 32, Batch_Size = 1
>
>
> (2) Settings for [CompVis/stable-diffusion-v1-4](./examples/huggingface/pytorch/text-to-image/deployment/stable_diffusion/): 
>
> Input = "a photo of an astronaut riding a horse on mars", Batch_Size = 1
>

## Documentation
<table>
<thead>
  <tr>
    <th colspan="8" align="center">OVERVIEW</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td colspan="2" align="center"><a href="docs">Model Compression</a></td>
    <td colspan="2" align="center"><a href="workflows/chatbot">NeuralChat</a></td>
    <td colspan="2" align="center"><a href="intel_extension_for_transformers/backends/neural_engine/docs">Neural Engine</a></td>
    <td colspan="2" align="center"><a href="intel_extension_for_transformers/backends/neural_engine/kernels/README.md">Kernel Libraries</a></td>
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
    <td colspan="2" align="center"><a href="intel_extension_for_transformers/backends/neural_engine/docs/onnx_compile.md">Model Compilation</a></td>
    <td colspan="2" align="center"><a href="intel_extension_for_transformers/backends/neural_engine/docs/add_customized_pattern.md">Custom Pattern</a></td>
    <td colspan="2" align="center"><a href="intel_extension_for_transformers/backends/neural_engine/docs/deploy_and_integration.md">Deployment</a></td>
    <td colspan="2" align="center"><a href="intel_extension_for_transformers/backends/neural_engine/docs/engine_profiling.md">Profiling</a></td>
  </tr>
  <tr>
    <th colspan="8" align="center">KERNEL LIBRARIES</th>
  </tr>
    <td colspan="2" align="center"><a href="intel_extension_for_transformers/backends/neural_engine/kernels/docs/kernel_desc">Sparse GEMM Kernels</a></td>
    <td colspan="2" align="center"><a href="intel_extension_for_transformers/backends/neural_engine/kernels/docs/kernel_desc">Custom INT8 Kernels</a></td>
    <td colspan="2" align="center"><a href="intel_extension_for_transformers/backends/neural_engine/kernels/docs/profiling.md">Profiling</a></td>
    <td colspan="2" align="center"><a href="intel_extension_for_transformers/backends/neural_engine/test/kernels/benchmark/benchmark.md">Benchmark</a></td>
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
    <td colspan="2" align="center"><a href="intel_extension_for_transformers/backends/neural_engine/docs/validated_model.md">Model Performance</a></td>
    <td colspan="2" align="center"><a href="intel_extension_for_transformers/backends/neural_engine/kernels/docs/validated_data.md">Kernel Performance</a></td>
  </tr>
</tbody>
</table>


## Selected Publications/Events
* Blog published on Medium: [Create Your Own Custom Chatbot](https://medium.com/intel-analytics-software/create-your-own-chatbot-on-cpus-b8d186cfefb2) (April 2023)
* Blog of Tech-Innovation Artificial-Intelligence(AI): [Intel® Xeon® Processors Are Still the Only CPU With MLPerf Results, Raising the Bar By 5x - Intel Communities](https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/Intel-Xeon-Processors-Are-Still-the-Only-CPU-With-MLPerf-Results/post/1472750) (April 2023)
* Blog published on Medium: [MLefficiency — Optimizing transformer models for efficiency](https://medium.com/@kawapanion/mlefficiency-optimizing-transformer-models-for-efficiency-a9e230cff051) (Dec 2022)
* NeurIPS'2022: [Fast Distilbert on CPUs](https://arxiv.org/abs/2211.07715) (Nov 2022)
* NeurIPS'2022: [QuaLA-MiniLM: a Quantized Length Adaptive MiniLM](https://arxiv.org/abs/2210.17114) (Nov 2022)
* Blog published by Cohere: [Top NLP Papers—November 2022](https://txt.cohere.ai/top-nlp-papers-november-2022/) (Nov 2022)
* Blog published by Alibaba: [Deep learning inference optimization for Address Purification](https://zhuanlan.zhihu.com/p/552484413) (Aug 2022)
* NeurIPS'2021: [Prune Once for All: Sparse Pre-Trained Language Models](https://arxiv.org/abs/2111.05754) (Nov 2021)
