# Quantization
Quantization is a widely-used model compression technique that can reduce model size while also improving inference and training latency. The full precision data converts to low-precision, there is little degradation in model accuracy, but the inference performance of quantized model can gain higher performance by saving the memory bandwidth and accelerating computations with low precision instructions. Intel provided several lower precision instructions (ex: 8-bit or 16-bit multipliers), both training and inference can get benefits from them. Refer to the Intel article on lower numerical precision inference and training in deep learning.

## Quantization Approach
### Post-Training Static Quantization performs quantization on already trained models, it requires an additional pass over the dataset to work, only activations do calibration.
<img src="imgs/PTQ.png" width=256 height=129 alt="PTQ">
<br>

### Post-Training Dynamic Quantization: Weights are quantized ahead of time but the activations are dynamically quantized during inference, the scale factor for activations dynamically based on the data range observed at runtime.
<img src="imgs/dynamic_quantization.png" width=270 height=124 alt="Dynamic Quantization">
<br>

### Quantization-aware Training (QAT) quantizes models during training and typically provides higher accuracy comparing with post-training quantization, but QAT may require additional hyper-parameter tuning and it may take more time to deployment.
<img src="imgs/QAT.png" width=244 height=147 alt="QAT">


## Quantization Usage
### Script:
```python
from intel_extension_for_transformers import metric, objectives, QuantizationConfig
from intel_extension_for_transformers.optimization.trainer import NLPTrainer
# Replace transformers.Trainer with NLPTrainer
# trainer = transformers.Trainer(......)
trainer = NLPTrainer(......)
metric = metrics.Metric(
    name="eval_f1", is_relative=True, criterion=0.01
)
objective = objectives.performance
q_config = QuantizationConfig(
    approach="PostTrainingStatic",
    metrics=[metric],
    objectives=[objective]
)
model = trainer.quantize(quant_config=q_config)
```
Please refer to [quantization example](../examples/optimize/pytorch/huggingface/text-classification/quantization/inc/run_glue.py) for the details.

### Create an instance of Metric
The Metric defines which metric will be used to measure the performance of tuned models.
- example:
    ```python
    metric = metrics.Metric(name="eval_f1", greater_is_better=True, is_relative=True, criterion=0.01, weight_ratio=None)
    ```

    Please refer to [metrics document](metrics.md) for the details.

### Create an instance of Objective(Optional)
In terms of evaluating the status of a specific model during tuning, we should have general objectives to measure the status of different models.

- example:
    ```python
    objective = objectives.Objective(name="performance", greater_is_better=True, weight_ratio=None)
    ```

    Please refer to [objective document](objectives.md) for the details.

### Create an instance of QuantizationConfig
The QuantizationConfig contains all the information related to the model quantization behavior. If you have created Metric and Objective instance(default Objective is "performance"), then you can create an instance of QuantizationConfig.

- arguments:
    |Argument   |Type       |Description                                        |Default value    |
    |:----------|:----------|:-----------------------------------------------|:----------------|
    |framework  |string     |Which framework you used                        |"pytorch"        |
    |approach   |string     |Which quantization approach you used            |"PostTrainingStatic"|
    |timeout    |integer    |Tuning timeout(seconds), 0 means early stop; combine with max_trials field to decide when to exit|0    |
    |max_trials |integer    |Max tune times                                  |100              |
    |metrics    |list of Metric|Used to evaluate accuracy of tuning model, no need for NoTrainerOptimizer|None |
    |objectives |list of Objective|Objective with accuracy constraint guaranteed|performance|

- example:
    ```python
    q_config = QuantizationConfig(
        approach="PostTrainingDynamic",
        metrics=[metric],
        objectives=[objective]
    )
    ```

### Quantization with Trainer
- Quantization with Trainer
    NLPTrainer inherits from transformers.Trainer, so you can create trainer like you do in transformers examples. Then you can quantize model with trainer.quantize function.
    ```python
    model = trainer.quantize(quant_config=q_config)
    ```
