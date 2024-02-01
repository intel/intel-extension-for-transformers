#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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

from intel_extension_for_transformers.llm.runtime.deprecated.compile.graph import Graph

def neural_engine_init(ir_path):
    text_encoder_graph = Graph()
    text_encoder_path = ir_path + '/text_encoder/'
    text_encoder_conf = text_encoder_path + 'conf.yaml'
    text_encoder_bin = text_encoder_path + 'model.bin'
    text_encoder_graph.graph_init(text_encoder_conf, text_encoder_bin)

    unet_graph = Graph()
    uent_path = ir_path + '/unet/'
    unet_conf = uent_path + 'conf.yaml'
    unet_bin = uent_path + 'model.bin'
    unet_graph.graph_init(unet_conf, unet_bin, True)

    vae_decoder_graph = Graph()
    vae_decoder_path = ir_path + '/vae_decoder/'
    vae_decoder_conf = vae_decoder_path + 'conf.yaml'
    vae_decoder_bin = vae_decoder_path + 'model.bin'
    vae_decoder_graph.graph_init(vae_decoder_conf, vae_decoder_bin)

    return [text_encoder_graph, unet_graph, vae_decoder_graph]
