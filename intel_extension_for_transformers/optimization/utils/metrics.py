#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The class of metric for optimization."""

class Metric(object):
    """Metric for optimization."""
    def __init__(self, name: str, greater_is_better: bool = True, is_relative: bool = True,
                 criterion: float = 0.01, weight_ratio: float = None):
        """Init an instance.

        Args:
            name: Metric name which evaluates function returns, like:"eval_f1", "eval_accuracy"...
            greater_is_better: Used to describe the usage of the metric, like: greater is better for f1, 
                this parameter is only used for quantization.
            is_relative: Used in conjunction with "criterion". If "criterion" is 0.01, and "is_relative" 
                is True, it means that we want to get an optimized model which metric drop <1% relative, 
                if "is_relative" is False, means metric drop <1% absolute, this parameter is only used 
                for quantization.
            criterion: Used in conjunction with "is_relative". If "criterion" is 0.01, and "is_relative" 
                is True, it means that we want to get an optimized model which metric drop <1% relative, 
                if "criterion" is 0.02, means metric drop <2% relative, this parameter is only used for 
                quantization.
            weight_ratio: Used when there are multiple metrics, for example: you want to focus on both 
                f1 and accuracy, then you will create f1 instance and accuracy instance, and indicate 
                their weight proportion. If weight_ratio of f1 is 0.3, and weight ratio of accuracy 
                is 0.7, then the final metric to tune is f1*0.3 + accuracy*0.7, this parameter is only 
                used for quantization.
        """
        self.name = name
        self.is_relative = is_relative
        self.criterion = criterion
        self.greater_is_better = greater_is_better
        self.weight_ratio = weight_ratio
