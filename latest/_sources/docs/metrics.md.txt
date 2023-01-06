Metrics
=======
1. [Introduction](#introduction)
2. [Supported Metric](#supported-metric)
3. [Metric Class Summary](#metric-class-summary)
4. [Get Start with Metrics](#get-start-with-metrics)

## Introduction
In terms of evaluating the performance of a specific model, we should have general metrics to measure the performance of different models. The Metric defines which metric will be used to measure the performance of tuned models and how to use the metric, just like: greater is better, performance tolerance ... and so on. The Metric only provides the metric name, all metrics class is from [datasets](https://github.com/huggingface/datasets/tree/main/metrics).

## Supported Metric
All metrics be provide by [Huggingface datasets](https://github.com/huggingface/datasets/tree/main/metrics).

## Metric Class Summary
- arguments:
    |Argument   |Type       |Description                                        |Default value    |
    |:----------|:----------|:-----------------------------------------------|:----------------|
    |name       |string     |Metric name which evaluates function returns, like:"eval_f1", "eval_accuracy"...|        |
    |greater_is_better|bool |Used to describe the usage of the metric, like: greater is better for f1, this parameter is only used for quantization.|True|
    |is_relative|bool       |Used in conjunction with "criterion". If "criterion" is 0.01, and "is_relative" is True, it means that we want to get an optimized model which metric drop <1% relative, if "is_relative" is False, means metric drop <1% absolute, this parameter is only used for quantization.|True    |
    |criterion  |float    |Used in conjunction with "is_relative". If "criterion" is 0.01, and "is_relative" is True, it means that we want to get an optimized model which metric drop <1% relative, if "criterion" is 0.02, means metric drop <2% relative, this parameter is only used for quantization.|0.01              |
    |weight_ratio|float   |Used when there are multiple metrics, for example: you want to focus on both f1 and accuracy, then you will create f1 instance and accuracy instance, and indicate their weight proportion. If weight_ratio of f1 is 0.3, and weight ratio of accuracy is 0.7, then the final metric to tune is f1*0.3 + accuracy*0.7, this parameter is only used for quantization.|None |

## Get Start with Metrics
- example:
    ```python
    from intel_extension_for_transformers.optimization import metric
    metric.Metric(name="eval_f1", greater_is_better=True, is_relative=True, criterion=0.01, weight_ratio=None)
    ```