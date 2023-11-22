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

from tqdm import tqdm
import torch
from torch import nn


class Audio2Exp(nn.Module):
    def __init__(self, netG, cfg, device, prepare_training_loss=False):
        super(Audio2Exp, self).__init__()
        self.cfg = cfg
        self.device = device
        self.netG = netG.to(device)

    def test(self, batch):
        mel_input = batch["indiv_mels"]  # bs T 1 80 16
        bs = mel_input.shape[0]
        T = mel_input.shape[1]

        exp_coeff_pred = []

        for i in tqdm(range(0, T, 10), "audio2exp:"):  # every 10 frames
            current_mel_input = mel_input[:, i : i + 10]

            # ref = batch['ref'][:, :, :64].repeat((1,current_mel_input.shape[1],1))           #bs T 64
            ref = batch["ref"][:, :, :64][:, i : i + 10]
            ratio = batch["ratio_gt"][:, i : i + 10]  # bs T

            audiox = current_mel_input.view(-1, 1, 80, 16)  # bs*T 1 80 16

            curr_exp_coeff_pred = self.netG(audiox, ref, ratio)  # bs T 64

            exp_coeff_pred += [curr_exp_coeff_pred]

        # BS x T x 64
        results_dict = {"exp_coeff_pred": torch.cat(exp_coeff_pred, axis=1)}
        return results_dict
