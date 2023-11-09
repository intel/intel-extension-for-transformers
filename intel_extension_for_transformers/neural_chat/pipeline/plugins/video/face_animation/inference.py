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

import shutil
import torch
from time import strftime
import os, sys, time
from argparse import ArgumentParser

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path

from datetime import datetime
import json


def main(args):
    import random

    random.seed(319)
    # torch.backends.cudnn.enabled = False

    pic_path = args.source_image
    audio_path = args.driven_audio
    save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)
    pose_style = args.pose_style
    device = args.device
    batch_size = args.batch_size
    input_yaw_list = args.input_yaw
    input_pitch_list = args.input_pitch
    input_roll_list = args.input_roll
    ref_eyeblink = args.ref_eyeblink
    ref_pose = args.ref_pose

    current_root_path = os.path.split(sys.argv[0])[0]

    sadtalker_paths = init_path(
        args.checkpoint_dir, os.path.join(current_root_path, "src/config"), args.size, args.old_version, args.preprocess
    )

    # init model
    timestamp = datetime.timestamp(datetime.now())
    print("start to generate video...", timestamp)
    preprocess_model = CropAndExtract(sadtalker_paths, device)
    start_time = time.time()
    audio_to_coeff = Audio2Coeff(sadtalker_paths, device)
    end_time = time.time()
    print("0000: Audio2Coeff")
    print(end_time - start_time)
    start_time = end_time
    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device, args.bf16)
    end_time = time.time()
    print("0001: AnimateFromCoeff")
    print(end_time - start_time)
    start_time = end_time
    # crop image and extract 3dmm from image
    first_frame_dir = os.path.join(save_dir, "first_frame_dir")
    os.makedirs(first_frame_dir, exist_ok=True)
    print("3DMM Extraction for source image")

    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
        pic_path, first_frame_dir, args.preprocess, source_image_flag=True, pic_size=args.size
    )
    end_time = time.time()
    print("0002: preprocess_model generate")
    print(end_time - start_time)
    start_time = end_time

    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return
    print("eyeblick? pose?")
    print(ref_eyeblink)
    print(ref_pose)
    if ref_eyeblink is not None:
        ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
        ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
        print("3DMM Extraction for the reference video providing eye blinking")
        ref_eyeblink_coeff_path, _, _ = preprocess_model.generate(
            ref_eyeblink, ref_eyeblink_frame_dir, args.preprocess, source_image_flag=False
        )
    else:
        ref_eyeblink_coeff_path = None

    if ref_pose is not None:
        if ref_pose == ref_eyeblink:
            ref_pose_coeff_path = ref_eyeblink_coeff_path
        else:
            ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
            ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
            os.makedirs(ref_pose_frame_dir, exist_ok=True)
            print("3DMM Extraction for the reference video providing pose")
            ref_pose_coeff_path, _, _ = preprocess_model.generate(
                ref_pose, ref_pose_frame_dir, args.preprocess, source_image_flag=False
            )
    else:
        ref_pose_coeff_path = None

    # audio2ceoff
    batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=args.still)
    coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)
    end_time = time.time()
    print("0003: audio_to_coeff generate...")
    print(end_time - start_time)
    start_time = end_time

    # coeff2video
    if args.rank == 0:
        data = get_facerender_data(
            coeff_path,
            crop_pic_path,
            first_coeff_path,
            audio_path,
            batch_size,
            input_yaw_list,
            input_pitch_list,
            input_roll_list,
            expression_scale=args.expression_scale,
            still_mode=args.still,
            preprocess=args.preprocess,
            size=args.size,
        )
        shutil.rmtree("workspace", ignore_errors=True)
        os.mkdir("workspace")
        # dict_keys(['source_image', 'source_semantics', 'frame_num',
        #           'target_semantics_list', 'video_name', 'audio_path'])
        torch.save(data["source_image"], "workspace/source_image.pt")
        torch.save(data["source_semantics"], "workspace/source_semantics.pt")
        torch.save(data["target_semantics_list"], "workspace/target_semantics_list.pt")
        meta = {}
        meta["frame_num"] = data["frame_num"]
        meta["video_name"] = data["video_name"]
        meta["audio_path"] = data["audio_path"]
        with open("workspace/meta.json", "w") as outfile:
            json.dump(meta, outfile)
    else:
        data = {}
        for pt_path in [
            "workspace/source_image.pt",
            "workspace/source_semantics.pt",
            "workspace/target_semantics_list.pt",
        ]:
            while os.path.exists(pt_path) == False:
                time.sleep(0.2)

            pkey = pt_path.split("/")[1].split(".")[0]
            try:
                data[pkey] = torch.load(pt_path)
            except:
                print("reload...")
                time.sleep(1)
                data[pkey] = torch.load(pt_path)
        while os.path.exists("workspace/meta.json") == False:
            time.sleep(0.2)
        with open("workspace/meta.json", "r") as read_content:
            meta = json.load(read_content)
            data["frame_num"] = meta["frame_num"]
            data["video_name"] = meta["video_name"]
            data["audio_path"] = meta["audio_path"]

    result = animate_from_coeff.generate(
        data,
        save_dir,
        pic_path,
        crop_info,
        enhancer=args.enhancer,
        background_enhancer=args.background_enhancer,
        preprocess=args.preprocess,
        img_size=args.size,
        rank=args.rank,
        p_num=args.p_num,
        bf16=args.bf16,
    )
    # os.remove('target_semantics.pt')
    shutil.rmtree("logs", ignore_errors=True)
    shutil.rmtree("enhancer_logs", ignore_errors=True)
    shutil.rmtree("workspace", ignore_errors=True)
    timestamp = datetime.timestamp(datetime.now())
    end_time = time.time()
    print("0004: render+enhance...")
    print(end_time - start_time)
    start_time = end_time
    print("generate video done...", timestamp)
    # shutil.move(result, save_dir+'.mp4')
    # print('The generated video is named:', save_dir+'.mp4')
    shutil.move(result, args.output_video_path)
    print("The generated video is named:", args.output_video_path)

    if not args.verbose:
        shutil.rmtree(save_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--driven_audio", default="./examples/driven_audio/bus_chinese.wav", help="path to driven audio"
    )
    parser.add_argument(
        "--source_image", default="./examples/source_image/full_body_1.png", help="path to source image"
    )
    parser.add_argument("--ref_eyeblink", default=None, help="path to reference video providing eye blinking")
    parser.add_argument("--ref_pose", default=None, help="path to reference video providing pose")
    parser.add_argument("--checkpoint_dir", default="./checkpoints", help="path to output")
    parser.add_argument("--result_dir", default="./results", help="path to output")
    parser.add_argument("--pose_style", type=int, default=0, help="input pose style from [0, 46)")
    parser.add_argument("--batch_size", type=int, default=2, help="the batch size of facerender")
    parser.add_argument("--size", type=int, default=256, help="the image size of the facerender")
    parser.add_argument("--expression_scale", type=float, default=1.0, help="the batch size of facerender")
    parser.add_argument("--input_yaw", nargs="+", type=int, default=None, help="the input yaw degree of the user ")
    parser.add_argument("--input_pitch", nargs="+", type=int, default=None, help="the input pitch degree of the user")
    parser.add_argument("--input_roll", nargs="+", type=int, default=None, help="the input roll degree of the user")
    parser.add_argument("--enhancer", type=str, default=None, help="Face enhancer, [gfpgan, RestoreFormer]")
    parser.add_argument("--background_enhancer", type=str, default=None, help="background enhancer, [realesrgan]")
    parser.add_argument("--cpu", dest="cpu", action="store_true")
    parser.add_argument("--face3dvis", action="store_true", help="generate 3d face and 3d landmarks")
    parser.add_argument(
        "--still", action="store_true", help="can crop back to the original videos for the full body aniamtion"
    )
    parser.add_argument(
        "--preprocess",
        default="crop",
        choices=["crop", "extcrop", "resize", "full", "extfull"],
        help="how to preprocess the images",
    )
    parser.add_argument("--verbose", action="store_true", help="saving the intermedia output or not")
    parser.add_argument("--old_version", action="store_true", help="use the pth other than safetensor version")

    # net structure and parameters
    parser.add_argument(
        "--net_recon", type=str, default="resnet50", choices=["resnet18", "resnet34", "resnet50"], help="useless"
    )
    parser.add_argument("--init_path", type=str, default=None, help="Useless")
    parser.add_argument("--use_last_fc", default=False, help="zero initialize the last fc")
    parser.add_argument("--bfm_folder", type=str, default="./checkpoints/BFM_Fitting/")
    parser.add_argument("--bfm_model", type=str, default="BFM_model_front.mat", help="bfm model")

    # default renderer parameters
    parser.add_argument("--focal", type=float, default=1015.0)
    parser.add_argument("--center", type=float, default=112.0)
    parser.add_argument("--camera_d", type=float, default=10.0)
    parser.add_argument("--z_near", type=float, default=5.0)
    parser.add_argument("--z_far", type=float, default=15.0)
    # distributed infer
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--p_num", type=int, default=1)
    # bf16
    parser.add_argument("--bf16", dest="bf16", action="store_true", help="whether to use bf16")
    # result video path
    parser.add_argument("--output_video_path", type=str, default="./response.mp4", help="the result video path")

    args = parser.parse_args()
    if torch.cuda.is_available() and not args.cpu:
        args.device = "cuda"
    else:
        args.device = "cpu"

    main(args)
