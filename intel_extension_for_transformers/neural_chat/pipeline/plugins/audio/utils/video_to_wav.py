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
import argparse
import subprocess
import shlex

from pydub import AudioSegment
import logging
logging.basicConfig(
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    datefmt="%d-%M-%Y %H:%M:%S",
    level=logging.INFO
)


def convert_video_to_wav(path, output_sample_rate, is_mono=True):
    path, basename = os.path.split(path)
    path_list = [basename]
    logging.info(path)

    output_dir = os.path.join(path, "../raw")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for filename in path_list:
        if os.path.isdir(os.path.join(path, filename)):
            continue
        filename_suffix = os.path.splitext(filename)[1]
        logging.info(filename)
        input_file_path = os.path.join(path, filename)
        output_file_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".wav")
        if filename_suffix == '.flv': # pragma: no cover
            sound = AudioSegment.from_flv(input_file_path)
            sound = sound.set_frame_rate(output_sample_rate)
            if is_mono:
                sound = sound.set_channels(1)
            sound.export(os.path.join(output_file_path), format="wav")
        elif filename_suffix == '.mp4' or filename_suffix == '.mp3':
            # file name should not contain space.
            if is_mono:
                cmd = "ffmpeg -i {} -ac 1 -ar {} -f wav {}".format(
                    input_file_path, output_sample_rate, output_file_path).split()
            else: # pragma: no cover
                cmd = "ffmpeg -i {} -ac 2 -ar {} -f wav {}".format(
                    input_file_path, output_sample_rate, output_file_path).split()
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e: # pragma: no cover
                logging.error("Error while executing command: %s", e)
        else: # pragma: no cover
            logging.info("file %s format not supported!", filename)
            continue


if __name__ == '__main__': # pragma: no cover
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--is_mono", type=str, default='True')
    parser.add_argument("--sr", type=str, default='16000')
    args = parser.parse_args()
    output_sample_rate = shlex.quote(args.sr)
    is_exist = os.path.exists(shlex.quote(args.path))
    if not is_exist:
        logging.info("path not existed!")
    else:
        path = shlex.quote(args.path)
        is_mono = shlex.quote(args.is_mono)
        convert_video_to_wav(path, output_sample_rate, is_mono)
