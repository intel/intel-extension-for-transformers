#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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

import sys
import os

common_dir = os.path.join(sys.path[0], "../../../../neural_engine_utils/")
sys.path.append(common_dir)
from common import (log, DummyDataLoader, compute_performance, Neural_Engine_base)


class Neural_Engine(Neural_Engine_base):
    def performance(self, batch_size, seq_len, iteration, warm_up):
        if warm_up >= iteration:
            log.error("Warm up should less than iteration.")
            raise ValueError()
        # generate dummy dataset
        log.info("Generate dummy dataset ......")
        shape = [batch_size, seq_len]
        dataset = DummyDataLoader(shapes=[shape, shape, shape],
                                  lows=[0, 0, 0],
                                  highs=[128, 1, 1],
                                  dtypes=['int32', 'int32', 'int32'],
                                  iteration=iteration)
        compute_performance(dataset, self.graph, log, self.log_file, warm_up, batch_size, seq_len)
