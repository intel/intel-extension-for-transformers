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

import os
import argparse
from datasets import load_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name',
                        help='dataset name',
                        type=str, default='glue')
    parser.add_argument('--task_name',
                        help='task name in dataset',
                        type=str, default='mrpc')
    parser.add_argument('--output_dir',
                        help='directory to save data to',
                        type=str, default='./data')
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    dataset = load_dataset(args.dataset_name, args.task_name,
                           cache_dir=args.output_dir, split='validation')

