# NLP Toolkit: Optimization for Natural Language Processing (NLP) Models
NLP Toolkit is a powerful toolkit for automatically applying model optimizations on Natural Language Processing Models. It leverages [Intel® Neural Compressor](https://intel.github.io/neural-compressor) to provide a variety of model compression techniques: quantization, pruning, distillation and so on.

## What does NLP Toolkit offer?
This toolkit allows developers to improve the productivity through ease-of-use model compression APIs by extending HuggingFace transformer APIs for deep learning models in NLP (Natural Language Processing) domain and accelerate the inference performance using compressed models.

- Model Compression

    |Framework          |Quantization |Pruning/Sparsity |Distillation |AutoDistillation |Length Adaptive |
    |-------------------|:-----------:|:---------------:|:-----------:|:---------------:|:--------------:|
    |PyTorch            |&#10004;     |&#10004;         |&#10004;     |&#10004;         |&#10004;        |
    |TensorFlow         |&#10004;     |&#10004;         |&#10004;     |Stay tuned :star:|                |

- Data Augmentation for NLP Datasets
- Neural Engine for Reference Deployment
- Sparse Lib for Sparse Reference Kernel

## Getting Started
### Installation
#### Install Dependency
```bash
pip install -r requirements.txt
```

#### Install NLP Toolkit
```bash
git clone https://github.com/intel-innersource/frameworks.ai.nlp-toolkit.intel-nlp-toolkit.git nlp_toolkit
cd nlp_toolkit
git submodule update --init --recursive
python setup.py install
```

### Quantization
```python
from nlp_toolkit import QuantizationConfig, metric, objectives
from nlp_toolkit.optimization.trainer import NLPTrainer

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
from nlp_toolkit import PrunerConfig, PruningConfig
from nlp_toolkit.optimization.trainer import NLPTrainer

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
from nlp_toolkit import DistillationConfig, Criterion
from nlp_toolkit.optimization.trainer import NLPTrainer

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
from nlp_toolkit.preprocessing.data_augmentation import DataAugmentation
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
from nlp_toolkit import QuantizationConfig, DynamicLengthConfig, metric, objectives
from nlp_toolkit.optimization.trainer import NLPTrainer

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


### Neural Engine
Neural Engine is one of reference deployments that NLP toolkit provides. Neural Engine aims to demonstrate the optimal performance of extremely compressed NLP models by exploring the optimization opportunities from both HW and SW.

```python
from nlp_toolkit.backends.neural_engine.compile import compile
# /path/to/your/model is a TensorFlow pb model or ONNX model
model = compile('/path/to/your/model')
inputs = ... # [input_ids, segment_ids, input_mask]
model.inference(inputs)
```

Please refer to [Neural Engine](examples/deployment/) for more details.

### Sparse Lib
SparseLib is a high-performance operator computing library implemented by assembly. SparseLib contains a JIT domain, a kernel domain, and a scheduling proxy framework.

```C++
#include "interface.hpp"
  ...
  operator_desc op_desc(ker_kind, ker_prop, eng_kind, ts_descs, op_attrs);
  sparse_matmul_desc spmm_desc(op_desc);
  sparse_matmul spmm_kern(spmm_desc);
  std::vector<const void*> rt_data = {data0, data1, data2, data3, data4};
  spmm_kern.execute(rt_data);
```

Please refer to [Sparse Lib](intel_extension_for_transformers/backends/neural_engine/SparseLib/README.md) for more details.
