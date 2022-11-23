# Intel® Extension for Transformers: Accelerating Transformer-based Models on Intel Platforms
Intel® Extension for Transformers is an innovative toolkit to accelerate Transformer-based models on Intel platforms. The toolkit helps developers to improve the productivity through ease-of-use model compression APIs by extending Hugging Face transformers APIs. The compression infrastructure leverages Intel® Neural Compressor which provides a rich set of model compression techniques: quantization, pruning, distillation and so on. The toolkit provides Transformers-accelerated Libraries and Neural Engine to demonstrate the performance of extremely compressed models, and therefore significantly improve the inference efficiency on Intel platforms. Some of the key features have been published in NeurIPS 2021 and 2022.

## What does Intel® Extension for Transformers offer?
This toolkit helps developers to improve the productivity of inference deployment by extending Hugging Face transformers APIs for Transformer-based models in natural language processing (NLP) domain. With extremely compressed models, the toolkit can greatly improve the inference efficiency on Intel platforms.

- Model Compression

    |Framework          |Quantization |Pruning/Sparsity |Distillation |Neural Architecture Search |
    |-------------------|:-----------:|:---------------:|:-----------:|:-------------------------:|
    |PyTorch            |&#10004;     |&#10004;         |&#10004;     |&#10004;                   |
    |TensorFlow         |&#10004;     |&#10004;         |&#10004;     |Stay tuned :star:          |

- Data Augmentation for NLP Datasets
- Transformers-accelerated Neural Engine
- Transformers-accelerated Libraries
- Domain Algorithms
    |Length Adaptive Transformer |
    |:--------------------------:|
    |PyTorch &#10004;            |

- Architecture of Intel® Extension for Transformers
<img src="docs/imgs/arch.png" width=691 height=444 alt="arch">
</br>

## Installation
#### Install Dependency
```bash
pip install -r requirements.txt
```

#### Install Intel® Extension for Transformers
```bash
git clone https://github.com/intel/intel-extension-for-transformers.git intel_extension_for_transformers
cd intel_extension_for_transformers
git submodule update --init --recursive
python setup.py install
```

## Getting Started
### Quantization
```python
from intel_extension_for_transformers import QuantizationConfig, metric, objectives
from intel_extension_for_transformers.optimization.trainer import NLPTrainer

# Replace transformers.Trainer with NLPTrainer
# trainer = transformers.Trainer(...)
trainer = NLPTrainer(...)
metric = metrics.Metric(name="eval_f1", is_relative=True, criterion=0.01)
q_config = QuantizationConfig(
    approach="PostTrainingStatic",
    metrics=[metric],
    objectives=[objectives.performance]
)
model = trainer.quantize(quant_config=q_config)
```

Please refer to [quantization document](docs/quantization.md) for more details.

### Pruning
```python
from intel_extension_for_transformers import PrunerConfig, PruningConfig
from intel_extension_for_transformers.optimization.trainer import NLPTrainer

# Replace transformers.Trainer with NLPTrainer
# trainer = transformers.Trainer(...)
trainer = NLPTrainer(...)
metric = metrics.Metric(name="eval_accuracy")
pruner_config = PrunerConfig(prune_type='BasicMagnitude', target_sparsity_ratio=0.9)
p_conf = PruningConfig(pruner_config=[pruner_config], metrics=metric)
model = trainer.prune(pruning_config=p_conf)
```

Please refer to [pruning document](docs/pruning.md) for more details.

### Distillation
```python
from intel_extension_for_transformers import DistillationConfig, Criterion
from intel_extension_for_transformers.optimization.trainer import NLPTrainer

# Replace transformers.Trainer with NLPTrainer
# trainer = transformers.Trainer(...)
teacher_model = ... # exist model
trainer = NLPTrainer(...)
metric = metrics.Metric(name="eval_accuracy")
d_conf = DistillationConfig(metrics=metric)
model = trainer.distill(distillation_config=d_conf, teacher_model=teacher_model)
```

Please refer to [distillation document](docs/distillation.md) for more details.

### Data Augmentation
Data augmentation provides the facilities to generate synthesized NLP dataset for further model optimization. The data augmentation supports text generation on popular fine-tuned models like GPT, GPT2, and other text synthesis approaches from [nlpaug](https://github.com/makcedward/nlpaug).

```python
from intel_extension_for_transformers.preprocessing.data_augmentation import DataAugmentation
aug = DataAugmentation(augmenter_type="TextGenerationAug")
aug.input_dataset = "original_dataset.csv" # example: https://huggingface.co/datasets/glue/viewer/sst2/train
aug.column_names = "sentence"
aug.output_path = os.path.join(self.result_path, "test2.cvs")
aug.augmenter_arguments = {'model_name_or_path': 'gpt2-medium'}
aug.data_augment()
raw_datasets = load_dataset("csv", data_files=aug.output_path, delimiter="\t", split="train")
```

Please refer to [data augmentation document](docs/data_augmentation.md) for more details.

### Quantized Length Adaptive Transformer
Quantized Length Adaptive Transformer leverages sequence-length reduction and low-bit representation techniques to further enhance model inference performance, enabling adaptive sequence-length sizes to accommodate different computational budget requirements with an optimal accuracy efficiency tradeoff.
```python
from intel_extension_for_transformers import QuantizationConfig, DynamicLengthConfig, metric, objectives
from intel_extension_for_transformers.optimization.trainer import NLPTrainer

# Replace transformers.Trainer with NLPTrainer
# trainer = transformers.Trainer(...)
trainer = NLPTrainer(...)
metric = metrics.Metric(name="eval_f1", is_relative=True, criterion=0.01)
q_config = QuantizationConfig(
    approach="PostTrainingStatic",
    metrics=[metric],
    objectives=[objectives.performance]
)
# Apply the length config
dynamic_length_config = DynamicLengthConfig(length_config=length_config)
trainer.set_dynamic_config(dynamic_config=dynamic_length_config)
# Quantization
model = trainer.quantize(quant_config=q_config)
```

Please refer to paper [QuaLA-MiniLM](https://arxiv.org/pdf/2210.17114.pdf) and [code](examples/optimization/pytorch/huggingface/question-answering/dynamic) for details


### Transformers-accelerated Neural Engine
Transformers-accelerated Neural Engine is one of reference deployments that Intel® Extension for Transformers provides. Neural Engine aims to demonstrate the optimal performance of extremely compressed NLP models by exploring the optimization opportunities from both HW and SW.

```python
from intel_extension_for_transformers.backends.neural_engine.compile import compile
# /path/to/your/model is a TensorFlow pb model or ONNX model
model = compile('/path/to/your/model')
inputs = ... # [input_ids, segment_ids, input_mask]
model.inference(inputs)
```

Please refer to [example](examples/deployment/neural_engine/sparse/distilbert_base_uncased/) in [Transformers-accelerated Neural Engine](examples/deployment/) and paper [Fast Distilbert on CPUs](https://arxiv.org/abs/2211.07715) for more details.

### Transformers-accelerated Libraries
Transformers-accelerated Libraries is a high-performance operator computing library implemented by assembly. Transformers-accelerated Libraries contains a JIT domain, a kernel domain, and a scheduling proxy framework.

```C++
#include "interface.hpp"
  ...
  operator_desc op_desc(ker_kind, ker_prop, eng_kind, ts_descs, op_attrs);
  sparse_matmul_desc spmm_desc(op_desc);
  sparse_matmul spmm_kern(spmm_desc);
  std::vector<const void*> rt_data = {data0, data1, data2, data3, data4};
  spmm_kern.execute(rt_data);
```
Please refer to [Transformers-accelerated Libraries](intel_extension_for_transformers/backends/neural_engine/Kernels/README.md) for more details.


## System Requirements
### Validated Hardware Environment
Intel® Extension for Transformers supports systems based on [Intel 64 architecture or compatible processors](https://en.wikipedia.org/wiki/X86-64) that are specifically optimized for the following CPUs:

* Intel Xeon Scalable processor (formerly Cascade Lake, Icelake)
* Future Intel Xeon Scalable processor (code name Sapphire Rapids)

### Validated Software Environment

* OS version: CentOS 8.4, Ubuntu 20.04  
* Python version: 3.7, 3.8, 3.9, 3.10  

<table class="docutils">
<thead>
  <tr>
    <th>Framework</th>
    <th>TensorFlow</th>
    <th>Intel TensorFlow</th>
    <th>PyTorch</th>
    <th>IPEX</th>
  </tr>
</thead>
<tbody>
  <tr align="center">
    <th>Version</th>
    <td class="tg-7zrl"><a href=https://github.com/tensorflow/tensorflow/tree/v2.10.0>2.10.0</a><br>
    <a href=https://github.com/tensorflow/tensorflow/tree/v2.9.1>2.9.1</a><br>
    <td class="tg-7zrl"><a href=https://github.com/Intel-tensorflow/tensorflow/tree/v2.10.0>2.10.0</a><br>
    <a href=https://github.com/Intel-tensorflow/tensorflow/tree/v2.9.1>2.9.1</a><br>
    <td class="tg-7zrl"><a href=https://download.pytorch.org/whl/torch_stable.html>1.13.0+cpu</a><br>
    <a href=https://download.pytorch.org/whl/torch_stable.html>1.12.0+cpu</a><br>
    <td class="tg-7zrl"><a href=https://github.com/intel/intel-extension-for-pytorch/tree/1.11.0>1.13.0</a><br>
    <a href=https://github.com/intel/intel-extension-for-pytorch/tree/v1.10.0>1.12.0</a></td>
  </tr>
</tbody>
</table>


## Selected Publications/Events
* NeurIPS'2022: [Fast Distilbert on CPUs](https://arxiv.org/abs/2211.07715) (Nov 2022)
* NeurIPS'2022: [QuaLA-MiniLM: a Quantized Length Adaptive MiniLM](https://arxiv.org/abs/2210.17114) (Nov 2022)
* Blog published by Alibaba: [Deep learning inference optimization for Address Purification](https://zhuanlan.zhihu.com/p/552484413) (Aug 2022)
* NeurIPS'2021: [Prune Once for All: Sparse Pre-Trained Language Models](https://arxiv.org/abs/2111.05754) (Nov 2021)
