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

"""The neural engine base extractor file.

   This is the base extractor class for tensorflow and onnx.
   Please refer to onnx_extractor.py and tf_extractor.py for more details.
"""

from .tf_extractor import TensorflowExtractor
from .onnx_extractor import ONNXExtractor
from .torch_extractor import TorchExtractor
from .. import logger
from ..graph_utils import get_model_fwk_name

EXTRACTORS = {
    'tensorflow': TensorflowExtractor,
    'onnxruntime': ONNXExtractor,
    'torch': TorchExtractor,
}


class Extractor(object):
    """Extractor base class.

    A super class for an operation extractor.
    Do additional extraction of operation attributes without modifying of graph topology.
    """

    def __call__(self, model, pattern_config = None):
        """The __call__ funtion of the base extractor class."""
        framework = model[1]
        extractor = EXTRACTORS[framework]()
        model = extractor(model[0])
        logger.info('Extract {} model done...'.format(framework))
        return model
