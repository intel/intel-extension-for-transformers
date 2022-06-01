# Quantization
## script:
```python
from nlp_toolkit import metric, NLPTrainer, objectives, QuantizationConfig,
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
Please refer to [quantization example](../examples/optimize/pytorch/huggingface/text-classification/quantization/inc/run_glue.py) for the details

## Create an instance of Metric
The Metric define which metric will used to measure the performance of tuned models.
- example:
    ```python
    metric = metrics.Metric(name="eval_f1", greater_is_better=True, is_relative=True, criterion=0.01, weight_ratio=None)
    ```

    Please refer to [metrics document](metrics.md) for the details.

## Create an instance of Objective(Optional)
In terms of evaluating the status of a specific model during tuning, we should have general objectives to measure the status of different models.

- example:
    ```python
    objective = objectives.Objective(name="performance", greater_is_better=True, weight_ratio=None)
    ```

    Please refer to [objective document](objectives.md) for the details.

## Create an instance of QuantizationConfig
The QuantizationConfig contains all the information related to the model quantization behavior. If you created Metric and Objective instance(default Objective is "performance"), then you can create an instance of QuantizationConfig.

- arguments:
    |Argument   |Type       |Description                                        |Default value    |
    |:----------|:----------|:-----------------------------------------------|:----------------|
    |framework  |string     |which framework you used                        |"pytorch"        |
    |approach   |string     |Which quantization approach you used            |"PostTrainingStatic"|
    |timeout    |integer    |Tuning timeout(seconds), 0 means early stop. combine with max_trials field to decide when to exit|0    |
    |max_trials |integer    |Max tune times                                  |100              |
    |metrics    |list of Metric|Used to evaluate accuracy of tuning model, no need for NoTrainerOptimizer|None |
    |objectives |list of Objective|objective with accuracy constraint guaranteed|performance|

- example:
    ```python
    q_config = QuantizationConfig(
        approach="PostTrainingDynamic",
        metrics=[metric],
        objectives=[objective]
    )
    ```

## Quantization with Trainer
- Quantization with Trainer
    NLPTrainer inherits from transformers.Trainer, so you can create trainer like you do in transformers examples. Then you can quantize model with trainer.quantize function.
    ```python
    from nlp_toolkit import metric, NLPTrainer, objectives, QuantizationConfig,
    trainer = NLPTrainer(......)
    model = trainer.quantize(quant_config=q_config)
    ```
