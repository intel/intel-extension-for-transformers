Pruning
======
1. [Introduction](#introduction)
2. [Pruning types](#pruning-types)
3. [Usage](#usage)

## Introduction
Pruning is the process of removing redundant parameters of a network. The idea bears similarity to the ["optimal brain damage"](http://yann.lecun.com/exdb/publis/pdf/lecun-90b.pdf) hypothesis by Yann LeCun. There are two types of pruning: Unstructured and Structured. Unstructured pruning means finding and removing the less salient connection in the model, the place could be anywhere in the matrix. Structured pruning means deleting entire blocks, filters, or channels.

## Pruning types

There are three pruning types in Intel® Extension for Transformers:

- Magnitude (Unstructured)
  - The algorithm prunes the weight by the lowest absolute value at each layer with a given sparsity target. 

- Group Lasso (Structured)
  - The algorithm uses Group lasso regularization to prune entire rows, columns, or blocks of parameters that result in a smaller dense network.

- Pattern Lock (Unstructured & Structured)
  - The algorithm locks the sparsity pattern in fine tune phase by freezing those zero values of the weight tensor during the weight update of training.

## Usage
### Script:
```python
from intel_extension_for_transformers.optimization import metric, objectives, PrunerConfig, PruningConfig,
from intel_extension_for_transformers.optimization.trainer import NLPTrainer
# Replace transformers.Trainer with NLPTrainer
# trainer = transformers.Trainer(......)
trainer = NLPTrainer(......)
metric = metrics.Metric(name="eval_accuracy")
pruner_config = PrunerConfig(prune_type='BasicMagnitude', target_sparsity_ratio=0.9)
p_conf = PruningConfig(pruner_config=[pruner_config], metrics=metric)
model = trainer.prune(pruning_config=p_conf)
```
Please refer to [example](../examples/huggingface/pytorch/text-classification/pruning) for the details.

### Create an instance of Metric
The Metric defines which metric will be used to measure the performance of tuned models.
- example:
    ```python
    metric = metrics.Metric(name="eval_accuracy")
    ```

    Please refer to [metrics document](metrics.md) for the details.

### Create list of an instance of PrunerConfig(Optional)
PrunerConfig defines which pruning algorithm to use and how to apply it during the training process. Intel® Extension for Transformers supports pruning types "BasicMagnitude", "PatternLock", and "GroupLasso". You can create different pruners for different layers.

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
    |parameters|dict of string|The hyper-parameters for pruning, refer to [the link](https://github.com/intel/neural-compressor/blob/master/docs/source/pruning.md)|None|

- example:
    ```python
    pruner_config = PrunerConfig(prune_type='BasicMagnitude', target_sparsity_ratio=0.9)
    ```

### Create an instance of PruningConfig
The PruningConfig contains all the information related to the model pruning behavior. If you have created Metric and PrunerConfig instance, then you can create an instance of PruningConfig. Metric and pruner are optional.

- arguments:
    |Argument   |Type       |Description                                        |Default value    |
    |:----------|:----------|:-----------------------------------------------|:----------------|
    |framework  |string     |Which framework you used                        |"pytorch"        |
    |initial_sparsity_ratio|float |Initial sparsity goal, if pruner_config argument is defined, it didn't need                       |0.0|
    |target_sparsity_ratio|float |Target sparsity goal, if pruner argument is defined, it didn't need                       |0.97|
    |metrics    |Metric    |Used to evaluate accuracy of tuning model, no need for NoTrainerOptimizer|None    |
    |pruner_config |PrunerConfig    |Defined pruning behavior, if it is None, then NLP will create a default a pruner with 'BasicMagnitude' pruning type                                  |None              |

- example:
    ```python
    pruning_conf = PruningConfig(pruner_config=[pruner_config], metrics=tune_metric)
    ```

### Prune with Trainer
- Prune with Trainer
    NLPTrainer inherits from `transformers.Trainer`, so you can create a trainer like what you do in transformers examples. Then you can prune model with `trainer.prune` function.
    ```python
    model = trainer.prune(pruning_config=pruning_conf)
    ```
