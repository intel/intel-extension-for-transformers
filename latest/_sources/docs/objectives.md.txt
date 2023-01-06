# Objective

1. [Introduction](#introduction)

1. [Supported Objectives Matrix](#supported-objectives-matrix)

1. [Examples](#examples)

## Introduction
In terms of evaluating the status of a specific model during tuning, we should have general objectives to measure the status of different models.

Intel Extension for Transformers supports optimized low-precision recipes for deep learning models to achieve optimal product objectives like inference performance and memory usage with expected accuracy criteria.

## Supported Objectives Matrix:
|Argument   |Type       |Description                                        |Default value    |
|:----------|:----------|:-----------------------------------------------|:----------------|
|name       |string     |a objective name in [Intel Neural Compressor](https://github.com/intel/neural-compressor/blob/master/docs/objective.md#built-in-objective-support-list). Like "performance", "modelsize",......and so on| / |
|greater_is_better|bool |used to describe the usage of the objective, like: greater is better for performance, but lower is better for modelsize| True |
|weight_ratio|float   |used when there are multiple objective. <br> for example: different weight proportion on performance and modelsize.| None |

## Examples:

There are two built-in objective instances: performance, modelsize. Users can also build their own objective as below:

```python
from intel_extension_for_transformers.objectives import performance, modelsize
```

or

```python
from intel_extension_for_transformers.optimization import objectives
performance = objectives.Objective(name="performance", greater_is_better=True, weight_ratio=None)
```
