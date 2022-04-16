# Metric
The Metric define which metric will used to measure the performance of tuned models.
- arguments:
    |Argument   |Type       |Description                                        |Default value    |
    |:----------|:----------|:-----------------------------------------------|:----------------|
    |name       |string     |Metric name which evaluate function returns, like:"eval_f1", "eval_accuracy"...|        |
    |greater_is_better|bool |Used to describe the usage of the metric, like: greater is better for f1, this parameter is only used for quantization.|True|
    |is_relative|bool       |Used in conjunction with "criterion", if "criterion" is 0.01, and "is_relative" is True, it means that we want to get an optimized model which metric drop <1% relative, if "is_relative" is False, means metric drop <1% absolute, this parameter is only used for quantization.|True    |
    |criterion  |float    |Used in conjunction with "is_relative". if "criterion" is 0.01, and "is_relative" is True, it means that we want to get an optimized model which metric drop <1% relative, if "criterion" is 0.02, means metric drop <2% relative, this parameter is only used for quantization.|0.01              |
    |weight_ratio|float   |Used when there are multiple metrics, for example: you want to focus on both F1 and accuracy, then you will create f1 instance and accuracy instance, and indicate their weight proportion. if weight_ratio of f1 is 0.3, and weight ratio of accuracy is 0.7, then the final metric to tune is f1*0.3 + accuracy*0.7, this parameter is only used for quantization.|None |

- example:
    ```python
    from nlp_toolkit import metric
    metric.Metric(name="eval_f1", greater_is_better=True, is_relative=True, criterion=0.01, weight_ratio=None)
    ```