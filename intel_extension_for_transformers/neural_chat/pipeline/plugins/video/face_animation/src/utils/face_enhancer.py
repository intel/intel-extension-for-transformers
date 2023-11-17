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

from gfpgan import GFPGANer

from tqdm import tqdm

from intel_extension_for_transformers.neural_chat.pipeline.plugins.video.face_animation.\
    src.utils.videoio import load_video_to_cv2

import cv2
import numpy as np
import contextlib
import time
import re
import itertools


class GeneratorWithLen(object):
    """From https://stackoverflow.com/a/7460929"""

    def __init__(self, gen, length):
        self.gen = gen
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.gen


def enhancer(images, method="gfpgan", bg_upsampler="realesrgan"):
    gen = enhancer_generator_no_len(images, method=method, bg_upsampler=bg_upsampler)
    return list(gen)


def enhancer_with_len(images, method="gfpgan", bg_upsampler="realesrgan", rank=0, p_num=1, bf16=False):
    if os.path.isfile(images):  # handle video to images
        # TODO: Create a generator version of load_video_to_cv2
        images = load_video_to_cv2(images)
        results = enhancer_no_len(images, method=method, bg_upsampler=bg_upsampler, rank=rank, p_num=p_num, bf16=bf16)
        # gen_with_len = GeneratorWithLen(gen, len(images))
        return results


def enhancer_no_len(images, method="gfpgan", bg_upsampler="realesrgan", rank=0, p_num=1, bf16=False):
    print(f"face enhancer rank, p_num: {rank}, {p_num}....")
    if not isinstance(images, list) and os.path.isfile(images):  # handle video to images
        images = load_video_to_cv2(images)

    # ------------------------ set up GFPGAN restorer ------------------------
    if method == "gfpgan":
        arch = "clean"
        channel_multiplier = 2
        model_name = "GFPGANv1.4"
        url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
    elif method == "RestoreFormer":
        arch = "RestoreFormer"
        channel_multiplier = 2
        model_name = "RestoreFormer"
        url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth"
    elif method == "codeformer":  # TODO:
        arch = "CodeFormer"
        channel_multiplier = 2
        model_name = "CodeFormer"
        url = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
    else:
        raise ValueError(f"Wrong model version {method}.")

    # set None to upsampler
    bg_upsampler = None

    # determine model paths
    model_path = os.path.join("gfpgan/weights", model_name + ".pth")

    if not os.path.isfile(model_path):
        model_path = os.path.join("checkpoints", model_name + ".pth")

    if not os.path.isfile(model_path):
        # download pre-trained models from url
        model_path = url

    restorer = GFPGANer(
        model_path=model_path, upscale=2, arch=arch, channel_multiplier=channel_multiplier, bg_upsampler=bg_upsampler
    )

    # ------------------------ restore ------------------------
    folder_name = "enhancer_logs"
    file_name = f"{p_num}_{rank}.npz"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    file_path = os.path.join(folder_name, file_name)
    r_imgs = []
    for idx in tqdm(range(len(images)), "Face Enhancer:"):
        if idx % p_num != rank:
            continue
        img = cv2.cvtColor(images[idx], cv2.COLOR_RGB2BGR)

        # restore faces and background if necessary
        # if bf16:
        with torch.cpu.amp.autocast(
            enabled=True, dtype=torch.bfloat16, cache_enabled=True
        ) if bf16 else contextlib.nullcontext():
            cropped_faces, restored_faces, r_img = restorer.enhance(  # r_img (512, 512, 3)
                img, has_aligned=False, only_center_face=False, paste_back=True
            )
        r_imgs.append(r_img)
    f = open(file_path, "w")
    np.savez(file_path, *r_imgs)
    f.close()

    ########## r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
    # yield r_img
    ######
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
        exit(0)
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
    aggregated_predictions = [i for i in padded_preds if i is not None]

    return [cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB) for r_img in aggregated_predictions]


def enhancer_generator_with_len(images, method="gfpgan", bg_upsampler="realesrgan"):
    """Provide a generator with a __len__ method so that it can passed to functions that
    call len()"""

    if os.path.isfile(images):  # handle video to images
        # TODO: Create a generator version of load_video_to_cv2
        images = load_video_to_cv2(images)

    gen = enhancer_generator_no_len(images, method=method, bg_upsampler=bg_upsampler)
    gen_with_len = GeneratorWithLen(gen, len(images))
    return gen_with_len


def enhancer_generator_no_len(images, method="gfpgan", bg_upsampler="realesrgan"):
    """Provide a generator function so that all of the enhanced images don't need
    to be stored in memory at the same time. This can save tons of RAM compared to
    the enhancer function."""

    print("face enhancer....")
    if not isinstance(images, list) and os.path.isfile(images):  # handle video to images
        images = load_video_to_cv2(images)

    # ------------------------ set up GFPGAN restorer ------------------------
    if method == "gfpgan":
        arch = "clean"
        channel_multiplier = 2
        model_name = "GFPGANv1.4"
        url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
    elif method == "RestoreFormer":
        arch = "RestoreFormer"
        channel_multiplier = 2
        model_name = "RestoreFormer"
        url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth"
    elif method == "codeformer":  # TODO:
        arch = "CodeFormer"
        channel_multiplier = 2
        model_name = "CodeFormer"
        url = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
    else:
        raise ValueError(f"Wrong model version {method}.")

    # set None to upsampler
    bg_upsampler = None

    # determine model paths
    model_path = os.path.join("gfpgan/weights", model_name + ".pth")

    if not os.path.isfile(model_path):
        model_path = os.path.join("checkpoints", model_name + ".pth")

    if not os.path.isfile(model_path):
        # download pre-trained models from url
        model_path = url

    restorer = GFPGANer(
        model_path=model_path, upscale=2, arch=arch, channel_multiplier=channel_multiplier, bg_upsampler=bg_upsampler
    )

    # ------------------------ restore ------------------------
    for idx in tqdm(range(len(images)), "Face Enhancer:"):
        img = cv2.cvtColor(images[idx], cv2.COLOR_RGB2BGR)

        # restore faces and background if necessary
        cropped_faces, restored_faces, r_img = restorer.enhance(
            img, has_aligned=False, only_center_face=False, paste_back=True
        )

        r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
        yield r_img
