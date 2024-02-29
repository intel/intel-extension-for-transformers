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
import uuid
import os
import cv2
import ffmpeg


def load_video_to_cv2(input_path):
    video_stream = cv2.VideoCapture(input_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    full_frames = []
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        full_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return full_frames


def save_video_with_watermark(video, audio, save_path, watermark=False):
    # temp_file = str(uuid.uuid4())+'.mp4'
    # cmd = r'ffmpeg -y -hide_banner -loglevel error -i "%s" -i "%s" -vcodec mpeg4 "%s"' % (video, audio, temp_file)
    # os.system(cmd)
    input_video = ffmpeg.input(video)
    input_audio = ffmpeg.input(audio)
    ffmpeg.concat(input_video, input_audio, v=1, a=1).output(
        save_path
    ).run()
