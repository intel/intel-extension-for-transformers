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
import cv2
import time
import glob
import argparse
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from itertools import cycle
from facexlib.alignment import init_alignment_model, landmark_98_to_68
from facexlib.detection import init_detection_model
from torch.multiprocessing import Pool, Process, set_start_method


class KeypointExtractor:
    def __init__(self, device="cuda"):
        root_path = "gfpgan/weights"

        print("---------device-----------", device)
        self.detector = init_alignment_model("awing_fan", device=device, model_rootpath=root_path)
        self.det_net = init_detection_model("retinaface_resnet50", half=False, device=device, model_rootpath=root_path)

    def extract_keypoint(self, images, name=None, info=True):
        if isinstance(images, list):
            keypoints = []
            if info:
                i_range = tqdm(images, desc="landmark Det:")
            else:
                i_range = images

            for image in i_range:
                current_kp = self.extract_keypoint(image)
                # current_kp = self.detector.get_landmarks(np.array(image))
                if np.mean(current_kp) == -1 and keypoints:
                    keypoints.append(keypoints[-1])
                else:
                    keypoints.append(current_kp[None])

            keypoints = np.concatenate(keypoints, 0)
            np.savetxt(os.path.splitext(name)[0] + ".txt", keypoints.reshape(-1))
            return keypoints
        else:
            while True:
                try:
                    with torch.no_grad():
                        # face detection -> face alignment.
                        img = np.array(images)
                        bboxes = self.det_net.detect_faces(images, 0.97)

                        bboxes = bboxes[0]
                        img = img[int(bboxes[1]) : int(bboxes[3]), int(bboxes[0]) : int(bboxes[2]), :]

                        keypoints = landmark_98_to_68(self.detector.get_landmarks(img))  # [0]

                        #### keypoints to the original location
                        keypoints[:, 0] += int(bboxes[0])
                        keypoints[:, 1] += int(bboxes[1])

                        break
                except RuntimeError as e:
                    if str(e).startswith("CUDA"):
                        print("Warning: out of memory, sleep for 1s")
                        time.sleep(1)
                    else:
                        print(e)
                        break
                except TypeError:
                    print("No face detected in this image")
                    shape = [68, 2]
                    keypoints = -1.0 * np.ones(shape)
                    break
            if name is not None:
                np.savetxt(os.path.splitext(name)[0] + ".txt", keypoints.reshape(-1))
            return keypoints


def read_video(filename):
    frames = []
    cap = cv2.VideoCapture(filename)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(frame)
        else:
            break
    cap.release()
    return frames


def run(data):
    filename, opt, device = data
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    kp_extractor = KeypointExtractor()
    images = read_video(filename)
    name = filename.split("/")[-2:]
    os.makedirs(os.path.join(opt.output_dir, name[-2]), exist_ok=True)
    kp_extractor.extract_keypoint(images, name=os.path.join(opt.output_dir, name[-2], name[-1]))


if __name__ == "__main__":
    set_start_method("spawn")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_dir", type=str, help="the folder of the input files")
    parser.add_argument("--output_dir", type=str, help="the folder of the output files")
    parser.add_argument("--device_ids", type=str, default="0,1")
    parser.add_argument("--workers", type=int, default=4)

    opt = parser.parse_args()
    filenames = list()
    VIDEO_EXTENSIONS_LOWERCASE = {"mp4"}
    VIDEO_EXTENSIONS = VIDEO_EXTENSIONS_LOWERCASE.union({f.upper() for f in VIDEO_EXTENSIONS_LOWERCASE})
    extensions = VIDEO_EXTENSIONS

    for ext in extensions:
        os.listdir(f"{opt.input_dir}")
        print(f"{opt.input_dir}/*.{ext}")
        filenames = sorted(glob.glob(f"{opt.input_dir}/*.{ext}"))
    print("Total number of videos:", len(filenames))
    pool = Pool(opt.workers)
    args_list = cycle([opt])
    device_ids = opt.device_ids.split(",")
    device_ids = cycle(device_ids)
    for data in tqdm(pool.imap_unordered(run, zip(filenames, args_list, device_ids))):
        None
