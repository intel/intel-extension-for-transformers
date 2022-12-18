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

"""The class of objective for optimization."""

class Objective(object):
    """Objective class."""
    def __init__(self, name: str, greater_is_better: bool = True, weight_ratio: float = None):
        """Init an instance.

        Args:
            name: Objectice name.
            greater_is_better: Used to describe the usage of the metric, like: greater is better for f1, 
                this parameter is only used for quantization.
            weight_ratio: Used when there are multiple metrics, for example: you want to focus on both 
                f1 and accuracy, then you will create f1 instance and accuracy instance, and indicate 
                their weight proportion. If weight_ratio of f1 is 0.3, and weight ratio of accuracy 
                is 0.7, then the final metric to tune is f1*0.3 + accuracy*0.7, this parameter is only 
                used for quantization.
        """
        self.name = name
        self.greater_is_better = greater_is_better
        self.weight_ratio = weight_ratio

    @staticmethod
    def performance():
        """Get a performance objective."""
        return Objective(name="performance", greater_is_better=True)

    @staticmethod
    def modelsize():
        """Get a modelsize objective."""
        return Objective(name="modelsize", greater_is_better=False)


performance = Objective(name="performance", greater_is_better=True)
modelsize = Objective(name="modelsize", greater_is_better=False)
