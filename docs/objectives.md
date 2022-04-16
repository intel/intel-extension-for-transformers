# Objective
In terms of evaluating the status of a specific model during tuning, we should have general objectives to measure the status of different models.

NLP Toolkit supports to optimized low-precision recipes for deep learning models to achieve optimal product objectives like inference performance and memory usage with expected accuracy criteria and now supports Objectives which is supported in [INC](https://github.com/intel-innersource/frameworks.ai.lpot.intel-lpot/blob/master/docs/objective.md#built-in-objective-support-list).

- arguments:
    |Argument   |Type       |Description                                        |Default value    |
    |:----------|:----------|:-----------------------------------------------|:----------------|
    |name       |string     |the Objective name in [INC](https://github.com/intel-innersource/frameworks.ai.lpot.intel-lpot/blob/master/docs/objective.md#built-in-objective-support-list). Like "performance", "modelsize",......and so on|        |
    |greater_is_better|bool |Used to describe the usage of the objective, like: greater is better for performance, but lower is better for modelsize|True|
    |weight_ratio|float   |Used when there are multiple objective, for example: you want to focus on both performance and modelsize, then you will create performance objective instance and modelsize objective instance, and indicate their weight proportion|None |

- example:
    ```python
    from nlp_toolkit import objectives
    objectives.Objective(name="performance", greater_is_better=True, weight_ratio=None)
    ```

- Built-in Objective instance: performance, modelsize.