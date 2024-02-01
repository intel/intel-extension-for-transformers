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

# pylint: disable=E0611
from scipy.spatial import ConvexHull
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import re, itertools
import contextlib


def normalize_kp(
    kp_source,
    kp_driving,
    kp_driving_initial,
    adapt_movement_scale=False,
    use_relative_movement=False,
    use_relative_jacobian=False,
):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source["value"][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial["value"][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = kp_driving["value"] - kp_driving_initial["value"]
        kp_value_diff *= adapt_movement_scale
        kp_new["value"] = kp_value_diff + kp_source["value"]

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving["jacobian"], torch.inverse(kp_driving_initial["jacobian"]))
            kp_new["jacobian"] = torch.matmul(jacobian_diff, kp_source["jacobian"])

    return kp_new


def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).type_as(pred).to(device)
    pred = F.softmax(pred)
    degree = torch.sum(pred * idx_tensor, 1) * 3 - 99
    return degree


def get_rotation_matrix(yaw, pitch, roll):
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    pitch_mat = torch.cat(
        [
            torch.ones_like(pitch),
            torch.zeros_like(pitch),
            torch.zeros_like(pitch),
            torch.zeros_like(pitch),
            torch.cos(pitch),
            -torch.sin(pitch),
            torch.zeros_like(pitch),
            torch.sin(pitch),
            torch.cos(pitch),
        ],
        dim=1,
    )
    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

    yaw_mat = torch.cat(
        [
            torch.cos(yaw),
            torch.zeros_like(yaw),
            torch.sin(yaw),
            torch.zeros_like(yaw),
            torch.ones_like(yaw),
            torch.zeros_like(yaw),
            -torch.sin(yaw),
            torch.zeros_like(yaw),
            torch.cos(yaw),
        ],
        dim=1,
    )
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

    roll_mat = torch.cat(
        [
            torch.cos(roll),
            -torch.sin(roll),
            torch.zeros_like(roll),
            torch.sin(roll),
            torch.cos(roll),
            torch.zeros_like(roll),
            torch.zeros_like(roll),
            torch.zeros_like(roll),
            torch.ones_like(roll),
        ],
        dim=1,
    )
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

    rot_mat = torch.einsum("bij,bjk,bkm->bim", pitch_mat, yaw_mat, roll_mat)

    return rot_mat


def keypoint_transformation(kp_canonical, he, wo_exp=False):
    kp = kp_canonical["value"]  # (bs, k, 3)
    yaw, pitch, roll = he["yaw"], he["pitch"], he["roll"]
    yaw = headpose_pred_to_degree(yaw)
    pitch = headpose_pred_to_degree(pitch)
    roll = headpose_pred_to_degree(roll)

    if "yaw_in" in he:
        yaw = he["yaw_in"]
    if "pitch_in" in he:
        pitch = he["pitch_in"]
    if "roll_in" in he:
        roll = he["roll_in"]

    rot_mat = get_rotation_matrix(yaw, pitch, roll)  # (bs, 3, 3)

    t, exp = he["t"], he["exp"]
    if wo_exp:
        exp = exp * 0

    # keypoint rotation
    kp_rotated = torch.einsum("bmp,bkp->bkm", rot_mat, kp)

    # keypoint translation
    t[:, 0] = t[:, 0] * 0
    t[:, 2] = t[:, 2] * 0
    t = t.unsqueeze(1).repeat(1, kp.shape[1], 1)
    kp_t = kp_rotated + t

    # add expression deviation
    exp = exp.view(exp.shape[0], -1, 3)
    kp_transformed = kp_t + exp

    return {"value": kp_transformed}


def make_animation(
    source_image,
    source_semantics,
    target_semantics,
    generator,
    kp_detector,
    he_estimator,
    mapping,
    yaw_c_seq=None,
    pitch_c_seq=None,
    roll_c_seq=None,
    use_exp=True,
    use_half=False,
    rank=0,
    p_num=1,
    bf16=False,
):
    print(f"rank, p_num: {rank}, {p_num}")
    with torch.no_grad():
        predictions = []
        import time
        import os

        # with torch.cpu.amp.autocast():
        start_time = time.time()
        kp_canonical = kp_detector(source_image)
        end_time = time.time()
        print("[kp_detector]:")
        print(end_time - start_time)
        start_time = end_time
        he_source = mapping(source_semantics)
        end_time = time.time()
        print("[mapping]:")
        print(end_time - start_time)
        start_time = end_time
        kp_source = keypoint_transformation(kp_canonical, he_source)
        end_time = time.time()
        print(end_time - start_time)
        for frame_idx in tqdm(range(target_semantics.shape[1]), "Face Renderer:"):
            if frame_idx % p_num != rank:
                continue
            target_semantics_frame = target_semantics[:, frame_idx]
            he_driving = mapping(target_semantics_frame)
            if yaw_c_seq is not None:
                he_driving["yaw_in"] = yaw_c_seq[:, frame_idx]
            if pitch_c_seq is not None:
                he_driving["pitch_in"] = pitch_c_seq[:, frame_idx]
            if roll_c_seq is not None:
                he_driving["roll_in"] = roll_c_seq[:, frame_idx]
            kp_driving = keypoint_transformation(kp_canonical, he_driving)
            kp_norm = kp_driving
            with torch.cpu.amp.autocast(
                enabled=True, dtype=torch.bfloat16, cache_enabled=True
            ) if bf16 else contextlib.nullcontext():
                out = generator(source_image, kp_source=kp_source, kp_driving=kp_norm)
                # print(f"{rank}:{frame_idx}")
                # print(out['prediction'])
                predictions.append(out["prediction"].to("cpu"))
        folder_name = "logs"
        file_name = f"{p_num}_{rank}.npz"
        os.makedirs(folder_name, exist_ok=True)
        file_path = os.path.join(folder_name, file_name)
        f = open(file_path, "w")
        np.savez(file_path, *predictions)
        f.close()

        # master process will be pending here to collect all the predictions
        # ... pending ...
        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(text):
            return [atoi(c) for c in re.split(r"(\d+)", text)]

        if rank == 0:
            while len(os.listdir(folder_name)) < p_num:
                time.sleep(0.2)
            # load all the npz arrays, merge by sequence
            # npz_file_paths = sorted(os.listdir(folder_name))
            npz_file_paths = os.listdir(folder_name)
            npz_file_paths.sort(key=natural_keys)
            print("start to merge...")
            print(f"npz_file_paths: {npz_file_paths}")
        else:
            # exit(0)
            return None
        aggregated_lst = []
        for npz_file_path in npz_file_paths:
            npz_file = np.load(os.path.join(folder_name, npz_file_path))
            aggregated_lst.append([npz_file[i] for i in npz_file.files])
        # aggregated_predictions = [torch.from_numpy(x) for y in zip(*aggregated_lst) for x in y]
        # agg lst elements may have different length!
        # exit(0)
        padded_preds = [x for y in itertools.zip_longest(*aggregated_lst) for x in y]
        print("padded preds length:")
        print(len(padded_preds))
        aggregated_predictions = [torch.from_numpy(i) for i in padded_preds if i is not None]
        # predictions_ts = torch.stack(predictions, dim=1)

        predictions_ts = torch.stack(aggregated_predictions, dim=1)
    return predictions_ts
