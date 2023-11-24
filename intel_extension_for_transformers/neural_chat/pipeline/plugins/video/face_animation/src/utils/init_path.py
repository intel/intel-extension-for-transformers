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

import os
import glob


def init_path(checkpoint_dir, config_dir, size=512, preprocess="crop"):
    """Init checkpoint paths."""
    print(os.path.join(checkpoint_dir, "*.safetensors"))
    if len(glob.glob(os.path.join(checkpoint_dir, "*.safetensors"))):
        print("using safetensor as default")
        sadtalker_paths = {
            "checkpoint": os.path.join(checkpoint_dir, "SadTalker_V0.0.2_" + str(size) + ".safetensors"),
        }
        use_safetensor = True
    else:
        raise Exception("Make Sure you download model checkpoints beforehand!")

    sadtalker_paths["dir_of_BFM_fitting"] = os.path.join(config_dir)  # , 'BFM_Fitting'
    sadtalker_paths["audio2pose_yaml_path"] = os.path.join(config_dir, "auido2pose.yaml")
    sadtalker_paths["audio2exp_yaml_path"] = os.path.join(config_dir, "auido2exp.yaml")
    sadtalker_paths["use_safetensor"] = use_safetensor  # os.path.join(config_dir, 'auido2exp.yaml')

    if "full" in preprocess:
        sadtalker_paths["mappingnet_checkpoint"] = os.path.join(checkpoint_dir, "mapping_00109-model.pth.tar")
        sadtalker_paths["facerender_yaml"] = os.path.join(config_dir, "facerender_still.yaml")
    else:
        sadtalker_paths["mappingnet_checkpoint"] = os.path.join(checkpoint_dir, "mapping_00229-model.pth.tar")
        sadtalker_paths["facerender_yaml"] = os.path.join(config_dir, "facerender.yaml")

    return sadtalker_paths
