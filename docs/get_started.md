# Getting Started

1. [Quantization](#quantization)

2. [Pruning](#pruning)

3. [Distillation](#distillation)

4. [Quantized Length Adaptive Transformer](#quantized-length-adaptive-transformer)

5. [Transformers-accelerated Neural Engine](#transformers-accelerated-neural-engine)


## Quantization
```python
from intel_extension_for_transformers.optimization import QuantizationConfig, metrics, objectives
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

For other Please refer to [quantization document](./quantization.md) for more details.


## Pruning
```python
from intel_extension_for_transformers.optimization import PrunerConfig, PruningConfig
from intel_extension_for_transformers.optimization.trainer import NLPTrainer

# Replace transformers.Trainer with NLPTrainer
# trainer = transformers.Trainer(...)
trainer = NLPTrainer(...)
metric = metrics.Metric(name="eval_accuracy")
pruner_config = PrunerConfig(prune_type='BasicMagnitude', target_sparsity_ratio=0.9)
p_conf = PruningConfig(pruner_config=[pruner_config], metrics=metric)
model = trainer.prune(pruning_config=p_conf)
```

Please refer to [pruning document](./pruning.md) for more details.


## Distillation
```python
from intel_extension_for_transformers.optimization import DistillationConfig, Criterion
from intel_extension_for_transformers.optimization.trainer import NLPTrainer

# Replace transformers.Trainer with NLPTrainer
# trainer = transformers.Trainer(...)
teacher_model = ... # exist model
trainer = NLPTrainer(...)
metric = metrics.Metric(name="eval_accuracy")
d_conf = DistillationConfig(metrics=metric)
model = trainer.distill(distillation_config=d_conf, teacher_model=teacher_model)
```

Please refer to [distillation document](./distillation.md) for more details.


## Quantized Length Adaptive Transformer
Quantized Length Adaptive Transformer leverages sequence-length reduction and low-bit representation techniques to further enhance model inference performance, enabling adaptive sequence-length sizes to accommodate different computational budget requirements with an optimal accuracy efficiency tradeoff.
```python
from intel_extension_for_transformers.optimization import QuantizationConfig, DynamicLengthConfig, metric, objectives
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

Please refer to paper [QuaLA-MiniLM](https://arxiv.org/pdf/2210.17114.pdf) and [code](../examples/optimization/pytorch/huggingface/question-answering/dynamic) for details


## Transformers-accelerated Neural Engine
Transformers-accelerated Neural Engine is one of reference deployments that Intel® Extension for Transformers provides. Neural Engine aims to demonstrate the optimal performance of extremely compressed NLP models by exploring the optimization opportunities from both HW and SW.

```python
from intel_extension_for_transformers.backends.neural_engine.compile import compile
# /path/to/your/model is a TensorFlow pb model or ONNX model
model = compile('/path/to/your/model')
inputs = ... # [input_ids, segment_ids, input_mask]
model.inference(inputs)
```

Please refer to [example](../examples/deployment/neural_engine/sparse/distilbert_base_uncased/) in [Transformers-accelerated Neural Engine](../examples/deployment/) and paper [Fast Distilbert on CPUs](https://arxiv.org/abs/2211.07715) for more details.