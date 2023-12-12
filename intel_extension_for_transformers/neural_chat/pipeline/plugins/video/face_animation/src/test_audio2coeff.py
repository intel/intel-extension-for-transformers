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
import torch
import numpy as np
from scipy.io import savemat, loadmat
from yacs.config import CfgNode as CN
from scipy.signal import savgol_filter

import safetensors
import safetensors.torch

from intel_extension_for_transformers.neural_chat.pipeline.plugins.video.face_animation.\
    src.audio2pose_models.audio2pose import Audio2Pose
from intel_extension_for_transformers.neural_chat.pipeline.plugins.video.face_animation.\
    src.audio2exp_models.networks import SimpleWrapperV2
from intel_extension_for_transformers.neural_chat.pipeline.plugins.video.face_animation.\
    src.audio2exp_models.audio2exp import Audio2Exp
from intel_extension_for_transformers.neural_chat.pipeline.plugins.video.face_animation.\
    src.utils.safetensor_helper import load_x_from_safetensor


def load_cpk(checkpoint_path, model=None, optimizer=None, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    if model is not None:
        model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["epoch"]


class Audio2Coeff:
    def __init__(self, sadtalker_path, device):
        # load config
        fcfg_pose = open(sadtalker_path["audio2pose_yaml_path"])
        cfg_pose = CN.load_cfg(fcfg_pose)
        cfg_pose.freeze()
        fcfg_exp = open(sadtalker_path["audio2exp_yaml_path"])
        cfg_exp = CN.load_cfg(fcfg_exp)
        cfg_exp.freeze()

        # load audio2pose_model
        self.audio2pose_model = Audio2Pose(cfg_pose, None, device=device)
        self.audio2pose_model = self.audio2pose_model.to(device)
        self.audio2pose_model.eval()
        for param in self.audio2pose_model.parameters():
            param.requires_grad = False

        if sadtalker_path["use_safetensor"]:
            checkpoints = safetensors.torch.load_file(sadtalker_path["checkpoint"])
            self.audio2pose_model.load_state_dict(load_x_from_safetensor(checkpoints, "audio2pose"))
        else:
            raise Exception("Make Sure you download model checkpoints beforehand!")

        # load audio2exp_model
        netG = SimpleWrapperV2()
        netG = netG.to(device)
        for param in netG.parameters():
            netG.requires_grad = False
        netG.eval()
        if sadtalker_path["use_safetensor"]:
            checkpoints = safetensors.torch.load_file(sadtalker_path["checkpoint"])
            netG.load_state_dict(load_x_from_safetensor(checkpoints, "audio2exp"))
        else:
            raise Exception("Make Sure you download model checkpoints beforehand!")

        self.audio2exp_model = Audio2Exp(netG, cfg_exp, device=device, prepare_training_loss=False)
        self.audio2exp_model = self.audio2exp_model.to(device)
        for param in self.audio2exp_model.parameters():
            param.requires_grad = False
        self.audio2exp_model.eval()

        self.device = device

    def generate(self, batch, coeff_save_dir, pose_style):
        with torch.no_grad():
            results_dict_exp = self.audio2exp_model.test(batch)
            exp_pred = results_dict_exp["exp_coeff_pred"]  # bs T 64

            batch["class"] = torch.LongTensor([pose_style]).to(self.device)
            results_dict_pose = self.audio2pose_model.test(batch)
            pose_pred = results_dict_pose["pose_pred"]  # bs T 6

            pose_len = pose_pred.shape[1]
            if pose_len < 13:
                pose_len = int((pose_len - 1) / 2) * 2 + 1
                pose_pred = torch.Tensor(savgol_filter(np.array(pose_pred.cpu()), pose_len, 2, axis=1)).to(self.device)
            else:
                pose_pred = torch.Tensor(savgol_filter(np.array(pose_pred.cpu()), 13, 2, axis=1)).to(self.device)

            coeffs_pred = torch.cat((exp_pred, pose_pred), dim=-1)  # bs T 70

            coeffs_pred_numpy = coeffs_pred[0].clone().detach().cpu().numpy()

            savemat(
                os.path.join(coeff_save_dir, "%s-%s.mat" % (batch["pic_name"], batch["audio_name"])),
                {"coeff_3dmm": coeffs_pred_numpy},
            )

            return os.path.join(coeff_save_dir, "%s-%s.mat" % (batch["pic_name"], batch["audio_name"]))

    def using_refpose(self, coeffs_pred_numpy, ref_pose_coeff_path):
        num_frames = coeffs_pred_numpy.shape[0]
        refpose_coeff_dict = loadmat(ref_pose_coeff_path)
        refpose_coeff = refpose_coeff_dict["coeff_3dmm"][:, 64:70]
        refpose_num_frames = refpose_coeff.shape[0]
        if refpose_num_frames < num_frames:
            div = num_frames // refpose_num_frames
            re = num_frames % refpose_num_frames
            refpose_coeff_list = [refpose_coeff for i in range(div)]
            refpose_coeff_list.append(refpose_coeff[:re, :])
            refpose_coeff = np.concatenate(refpose_coeff_list, axis=0)

        #### relative head pose
        coeffs_pred_numpy[:, 64:70] = coeffs_pred_numpy[:, 64:70] + (
            refpose_coeff[:num_frames, :] - refpose_coeff[0:1, :]
        )
        return coeffs_pred_numpy
