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

from ..graph_utils import LazyImport, get_model_fwk_name

onnx = LazyImport('onnx')
tf = LazyImport('tensorflow')

class Loader(object):
    def __call__(self, model):
        framework = get_model_fwk_name(model)
        if framework =='tensorflow':
            if isinstance(model, str):
                graph = tf.Graph()
                graph_def = tf.compat.v1.GraphDef()
                with open(model, 'rb') as f:
                    graph_def.ParseFromString(f.read())
                    with graph.as_default():
                        tf.import_graph_def(graph_def, name='')
                config = tf.compat.v1.ConfigProto()
                model = tf.compat.v1.Session(graph=graph, config=config)
        if framework =='onnxruntime':
            if isinstance(model, str):
                model = onnx.load(model)
        return model, framework
