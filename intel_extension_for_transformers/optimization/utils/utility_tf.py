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

"""Utils for tensorflow framework."""

import os
import json
from collections import OrderedDict, UserDict
from neural_compressor.experimental import common

TMPPATH = os.path.join('tmp', 'model')
TEACHERPATH = os.path.join('tmp', 'teacher_model')
class TFDataloader(object):
    """Tensorflow dataloader.

    Args:
        dataset (string): Dataset
    """

    def __init__(self, dataset, batch_size=None):
        """Init an instance."""
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        """Get the iteration of dataset."""
        for inputs, labels in self.dataset:
            if isinstance(inputs, dict) or isinstance(inputs, OrderedDict) \
                  or isinstance(inputs, UserDict):
                for name in inputs.keys():
                    inputs[name] = inputs[name].numpy()
            elif isinstance(inputs, list) or isinstance(inputs, tuple):
                inputs = [input.numpy() for input in inputs]
            else:
                inputs = inputs.numpy()

            if isinstance(labels, dict) or isinstance(labels, OrderedDict) \
                  or isinstance(labels, UserDict):   # pragma: no cover
                for name in labels.keys():
                    labels[name] = labels[name].numpy()
            elif isinstance(labels, list) or isinstance(labels, tuple):
                labels = [label.numpy() for label in labels]
            else:
                labels = labels.numpy()
            yield inputs, labels

    def __len__(self):
        """Return the length of dataset."""
        return len(self.dataset)


def distributed_init(worker_addresses, type='worker', index=0):
    """Init distribute environment.

    Args:
        worker_addresses: Addresses of all nodes.
        type: The type of node, such as worker.
        index: When index is 0, the node treat as a chief.
    """
    tf_config = {
        'cluster': {
            'worker': worker_addresses
        },
        'task': {'type': type, 'index': index}
    }
    os.environ['TF_CONFIG'] = json.dumps(tf_config)

def _is_chief(task_type, task_id):
    # here only consider the case in which TF_CONFIG task_type is set as worker
    # and task_id=0 represents the chief
    return (task_type == 'worker' and task_id == 0)

# get model folder path for the distributed environment
def get_filepath(base_dirpath, task_type, task_id):
    """Get model folder path for the distributed environment.

    Args:
        base_dirpath: The basic folder path.
        task_type: Task_type is set as worker.
        task_id: Task id. When task_id=0, the node treat as a chief.
    """
    if task_type is None:    # single node
        return base_dirpath
    elif _is_chief(task_type, task_id):
        return os.path.join(base_dirpath, 'chief')
    else:
        return os.path.join(base_dirpath, 'worker_' + str(task_id))


# convert a Keras model to SavedModel
def keras2SavedModel(model):   # pragma: no cover
    """Transfer keras model into save_model."""
    model = common.Model(model)
    return model.model