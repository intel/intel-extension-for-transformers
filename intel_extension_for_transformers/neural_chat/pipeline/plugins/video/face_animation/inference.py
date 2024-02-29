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
import logging
logging.basicConfig(
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    datefmt="%d-%M-%Y %H:%M:%S",
    level=logging.INFO
)

from intel_extension_for_transformers.neural_chat.pipeline.plugins.video.face_animation.\
    src.utils.preprocess import CropAndExtract
from intel_extension_for_transformers.neural_chat.pipeline.plugins.video.face_animation.\
    src.test_audio2coeff import Audio2Coeff
from intel_extension_for_transformers.neural_chat.pipeline.plugins.video.face_animation.\
    src.facerender.animate import AnimateFromCoeff
from intel_extension_for_transformers.neural_chat.pipeline.plugins.video.face_animation.\
    src.generate_batch import get_data
from intel_extension_for_transformers.neural_chat.pipeline.plugins.video.face_animation.\
    src.generate_facerender_batch import get_facerender_data
from intel_extension_for_transformers.neural_chat.pipeline.plugins.video.face_animation.\
    src.utils.init_path import init_path

from datetime import datetime
import json
import random

def main(args):
    all_start_timestamp = datetime.timestamp(datetime.now())
    random.seed(319)

    pic_path = args.source_image
    audio_path = args.driven_audio
    save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)
    pose_style = args.pose_style
    device = args.device
    batch_size = args.batch_size

    current_root_path = os.path.split(sys.argv[0])[0]

    sadtalker_paths = init_path(
        args.checkpoint_dir, os.path.join(current_root_path, "src/config"), args.size, args.preprocess
    )

    # init model
    preprocess_model = CropAndExtract(sadtalker_paths, device)
    start_time = time.time()
    audio_to_coeff = Audio2Coeff(sadtalker_paths, device)
    end_time = time.time()
    logging.info("[***1/6***]: Audio2Coeff takes: %s sec", end_time - start_time)
    start_time = end_time
    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device, args.bf16)
    end_time = time.time()
    logging.info("[***2/6***]: AnimateFromCoeff takes: %s sec", end_time - start_time)
    start_time = end_time
    # crop image and extract 3dmm from image
    first_frame_dir = os.path.join(save_dir, "first_frame_dir")
    os.makedirs(first_frame_dir, exist_ok=True)
    logging.info("3DMM Extraction for source image")

    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
        pic_path, first_frame_dir, args.preprocess, source_image_flag=True, pic_size=args.size
    )
    end_time = time.time()
    logging.info("[***3/6***]: preprocess_model.generate takes: %s sec", end_time - start_time)
    start_time = end_time

    if first_coeff_path is None:
        logging.info("Can't get the coeffs of the input")
        return

    # audio2coeff
    batch = get_data(first_coeff_path, audio_path, device, still=args.still)
    coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style)
    end_time = time.time()
    logging.info("[***4/6***]: audio_to_coeff takes: %s sec", end_time - start_time)
    start_time = end_time

    # coeff2video
    if args.rank == 0:
        data = get_facerender_data(
            coeff_path,
            crop_pic_path,
            first_coeff_path,
            audio_path,
            batch_size,
            expression_scale=args.expression_scale,
            still_mode=args.still,
            preprocess=args.preprocess,
            size=args.size,
        )
        shutil.rmtree("workspace", ignore_errors=True)
        os.mkdir("workspace")
        # dict_keys([
        # 'source_image', 'source_semantics', 'frame_num', 'target_semantics_list', 'video_name', 'audio_path'
        # ])
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
                logging.info("reload...")
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

    # gc all intermediate logs
    shutil.rmtree("logs", ignore_errors=True)
    shutil.rmtree("enhancer_logs", ignore_errors=True)
    shutil.rmtree("workspace", ignore_errors=True)
    shutil.move(result, args.output_video_path)

    logging.info("The generated video is named: %s", args.output_video_path)

    if not args.verbose:
        # print(save_dir)
        # shutil.rmtree(save_dir)
        shutil.rmtree(args.result_dir, ignore_errors=True)
    logging.info("Face animation done: %s sec", datetime.timestamp(datetime.now()) - all_start_timestamp)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--driven_audio", default="./examples/driven_audio/bus_chinese.wav", help="path to driven audio"
    )
    parser.add_argument(
        "--source_image", default="./examples/source_image/full_body_1.png", help="path to source image"
    )
    parser.add_argument("--checkpoint_dir", default="./checkpoints", help="path to output")
    parser.add_argument("--result_dir", default="./results", help="path to output")
    parser.add_argument("--pose_style", type=int, default=0, help="input pose style from [0, 46)")
    parser.add_argument("--batch_size", type=int, default=2, help="the batch size of facerender")
    parser.add_argument("--size", type=int, default=256, help="the image size of the facerender")
    parser.add_argument("--expression_scale", type=float, default=1.0, help="the expression scale of facerender")
    parser.add_argument("--enhancer", type=str, default=None, help="Face enhancer, [gfpgan, RestoreFormer]")
    parser.add_argument("--background_enhancer", type=str, default=None, help="background enhancer, [realesrgan]")
    parser.add_argument("--cpu", dest="cpu", action="store_true")
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
