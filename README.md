# NLP Toolkit：Optimization for Natural Language Processing (NLP) Models
NLP Toolkit is a powerful toolkit for automatically applying model optimizations on Natural Language Processing Models. It leverages [Intel® Neural Compressor](https://intel.github.io/neural-compressor) to provide a variety of optimization methods: quantization, pruning, distillation and so on.

## What does NLP Toolkit offer?
This toolkit allows developers to improve the productivity through ease-of-use model compression APIs by extending HuggingFace transformer APIs for deep learning models in NLP (Natural Language Processing) domain and accelerate the inference performance using compressed models.

- Model Compression

    |Framework          |Quantization          |Pruning/Sparsity |Distillation          |
    |-------------------|:--------------------:|:---------------:|:--------------------:|
    |PyTorch            |&#10004;              |&#10004;         |&#10004;              |

- Data Augmentation for NLP Datasets
- NLP Executor for Inference Acceleration

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
from nlp_toolkit import NLPTrainer, QuantizationConfig, metric, objectives

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
from nlp_toolkit import NLPTrainer, PrunerConfig, PruningConfig

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
from nlp_toolkit import NLPTrainer, DistillationConfig, Criterion

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

### NLP Executor
NLP Executor is an inference executor for Natural Language Processing (NLP) models, providing the optimal performance by quantization and sparsity. The executor is a baremetal reference engine of NLP Toolkit and supports typical NLP models.

```python
from engine.compile import compile
# /path/to/your/model is a TensorFlow pb model or ONNX model
model = compile('/path/to/your/model')
inputs = ... # [input_ids, segment_ids, input_mask]
model.inference(inputs)
```

Please refer to [NLP executor document](docs/nlp_executor.md) for more details.

