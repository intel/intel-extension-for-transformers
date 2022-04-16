# Pruning
## script:
```python
from nlp_toolkit import metric, NLPTrainer, objectives, QuantizationConfig,
# Create a trainer like you do in transformers examples, just replace transformers.Trainer with NLPTrainer
# ~~trainer = transformers.Trainer(......)~~
trainer = NLPTrainer(......)
tune_metric = metrics.Metric(name="eval_accuracy")
pruner = Pruner(prune_type='BasicMagnitude', target_sparsity_ratio=0.9)
pruning_conf = PruningConfig(pruner=[pruner], metrics=tune_metric)
model = trainer.prune(pruning_config=pruning_conf)
```
Please refer to [example](../examples/optimize/pytorch/huggingface/text-classification/pruning/run_glue.py) for the details.

## Create an instance of Metric
The Metric define which metric will used to measure the performance of tuned models.
- example:
    ```python
    Metric(name="eval_accuracy")
    ```

    Please refer to [metrics document](metrics.md) for the details.

## Create list of an instance of Pruner(Optional)
Pruner defines which pruning algorithm is used and how to apply it during training process. NLP Toolkit supports pruning type is "BasicMagnitude", "PatternLock", and "GroupLasso". You can create different pruner for different layers.

- arguments:
    |Argument   |Type       |Description                                        |Default value    |
    |:----------|:----------|:-----------------------------------------------|:----------------|
    |epoch_range|list of integer|Which epochs to pruning                     |[0, 4]           |
    |initial_sparsity_ratio|float |Initial sparsity goal                     |0.0              |
    |target_sparsity_ratio|float  |Target sparsity goal                      |0.97             |
    |update_frequency|integer|Frequency to updating sparsity                 |1                |
    |prune_type|string|Pruning algorithm                                     |'BasicMagnitude' |
    |method|string|Pruning method                                            |'per_tensor' |
    |names|list of string|List of weight name to be pruned. If no weight is specified, all weights of the model will be pruned|[]|
    |parameters|dict of string|The hyper-parameters for pruning, refer to [the link](https://github.com/intel/neural-compressor/blob/master/docs/pruning.md)|None|

- example:
    ```python
    Pruner(prune_type='BasicMagnitude', target_sparsity_ratio=0.9)
    ```

## Create an instance of PruningConfig
The PruningConfig contains all the information related to the model pruning behavior. If you created Metric and Pruner instance, then you can create an instance of PruningConfig. Metric and pruner is optional.

- arguments:
    |Argument   |Type       |Description                                        |Default value    |
    |:----------|:----------|:-----------------------------------------------|:----------------|
    |framework  |string     |which framework you used                        |"pytorch"        |
    |initial_sparsity_ratio|float |Initial sparsity goal, if pruner argument is defined, it didn't need                       |0.0|
    |target_sparsity_ratio|float |target sparsity goal, if pruner argument is defined, it didn't need                       |0.97|
    |metrics    |Metric    |Used to evaluate accuracy of tuning model, no need for NoTrainerOptimizer|None    |
    |pruner |Pruner    |Defined pruning behavior, if it is None, then NLP will create a default a pruner with 'BasicMagnitude' pruning type                                  |None              |

- example:
    ```python
    pruning_conf = PruningConfig(pruner=[pruner], metrics=tune_metric)
    ```

## Prune with Trainer
- Prune with Trainer
    NLPTrainer inherits from transformers.Trainer, so you can create trainer like you do in transformers examples. Then you can quantize model with trainer.prune function.
    ```python
    from nlp_toolkit import metric, NLPTrainer, objectives, QuantizationConfig,
    trainer = NLPTrainer(......)
    model = trainer.prune(pruning_config=pruning_conf)
    ```
