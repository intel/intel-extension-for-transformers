# Distillation
## script:
```python
from nlp_toolkit import metric, NLPTrainer, objectives, QuantizationConfig, Criterion
# Create a trainer like you do in transformers examples, just replace transformers.Trainer with NLPTrainer
# ~~trainer = transformers.Trainer(......)~~
trainer = NLPTrainer(......)
tune_metric = metrics.Metric(name="eval_accuracy")
distillation_conf = DistillationConfig(metrics=tune_metric)
model = trainer.distill(
    distillation_config=distillation_conf, teacher_model=teacher_model
)
```

Please refer to [example](../examples/optimize/pytorch/huggingface/text-classification/distillation/run_glue.py) for the details.

## Create an instance of Metric
The Metric define which metric will used to measure the performance of tuned models.
- example:
    ```python
    Metric(name="eval_accuracy")
    ```

    Please refer to [metrics document](metrics.md) for the details.

## Create an instance of Criterion(Optional)
The criterion used in training phase.

- arguments:
    |Argument   |Type       |Description                                        |Default value    |
    |:----------|:----------|:-----------------------------------------------|:----------------|
    |name       |String|Name of criterion, like:"KnowledgeLoss", "IntermediateLayersLoss"  |"KnowledgeLoss"|
    |temperature|Float |parameter for KnowledgeDistillationLoss               |1.0             |
    |loss_types|List of string|Type of loss                               |['CE', 'CE']        |
    |loss_weight_ratio|List of float|weight ratio of loss                 |[0.5, 0.5]     |
    |layer_mappings|List|parameter for IntermediateLayersLoss             |[] |
    |add_origin_loss|bool|parameter for IntermediateLayersLoss            |False |

- example:
    ```python
    Criterion(name='KnowledgeLoss')
    ```

## Create an instance of DistillationConfig
The DistillationConfig contains all the information related to the model distillation behavior. If you created Metric and Criterion instance, then you can create an instance of DistillationConfig. Metric and pruner is optional.

- arguments:
    |Argument   |Type       |Description                                        |Default value    |
    |:----------|:----------|:-----------------------------------------------|:----------------|
    |framework  |string     |which framework you used                        |"pytorch"        |
    |criterion|Criterion |criterion of training                              |"KnowledgeLoss"|
    |metrics    |Metric    |Used to evaluate accuracy of tuning model, no need for NoTrainerOptimizer|None    |

- example:
    ```python
    distillation_conf = DistillationConfig(metrics=tune_metric)
    ```

## Distill with Trainer
- Distill with Trainer
    NLPTrainer inherits from transformers.Trainer, so you can create trainer like you do in transformers examples. Then you can distill model with trainer.distill function.
    ```python
    from nlp_toolkit import metric, NLPTrainer, objectives, QuantizationConfig,
    trainer = NLPTrainer(......)
    model = trainer.distill(
        distillation_config=distillation_conf, teacher_model=teacher_model
    )
    ```