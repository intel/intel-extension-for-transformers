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

import subprocess
import os
import argparse
import shlex
import logging
logging.basicConfig(
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    datefmt="%d-%M-%Y %H:%M:%S",
    level=logging.INFO
)
def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    # print("Total seconds:", float(result.stdout))
    return float(result.stdout)

def cut_video(args, outdir):
    path = shlex.quote(args.path)
    save_path = outdir
    path, basename = os.path.split(path)
    video_list = [basename]
    delta_X = int(shlex.quote(args.min))

    for file_name in video_list:
        min = int(get_length(os.path.join(path, file_name))) // 60
        t_second = int(get_length(os.path.join(path, file_name))) % 60
        t_hour = min // 60
        t_min = min % 60
        logging.info("total: %s hours, %s minutes, %s seconds", t_hour, t_min, t_second)

        start_hour = 0
        start_min = 0
        start_sec = 0
        end_hour = 0
        end_min = 0
        end_sec = 0
        mark = 0

        for i in range(0, min + 1, delta_X):
            logging.info("i: %s", i)
            if min >= delta_X:
                end_min = start_min + delta_X
                end_sec = start_sec
                end_hour = start_hour

                if end_min >= 60:
                    end_hour = start_hour + end_min // 60
                    end_min = end_min % 60

                if end_hour > t_hour:
                    end_hour = t_hour
                if end_min > t_min and end_hour >= t_hour:
                    end_min = t_min

                if i == t_min:
                    end_sec = t_second

                start_hour = '0' + str(start_hour) if len(str(start_hour)) == 1 else str(start_hour)
                start_min = '0' + str(start_min) if len(str(start_min)) == 1 else str(start_min)
                start_sec = '0' + str(start_sec) if len(str(start_sec)) == 1 else str(start_sec)
                end_hour = '0' + str(end_hour) if len(str(end_hour)) == 1 else str(end_hour)
                end_min = '0' + str(end_min) if len(str(end_min)) == 1 else str(end_min)
                end_sec = '0' + str(end_sec) if len(str(end_sec)) == 1 else str(end_sec)

                name, _ = os.path.splitext(file_name)
                name = str(name) + "_" + str(mark)
                mark += 1
                command = 'ffmpeg -i {} -ss {}:{}:{} -to {}:{}:{} -ac 1 -ar {} -f wav {}'.format(
                    os.path.join(path,file_name), start_hour, start_min, start_sec, end_hour,
                    end_min, end_sec, shlex.quote(args.sr), os.path.join(save_path, str(name))+'.wav').split()
                logging.info("%s hours, %s minutes, %s seconds", start_hour, start_min, start_sec)
                logging.info("%s hours, %s minutes, %s seconds", end_hour, end_min, end_sec)
                try:
                    subprocess.run(command, check=True)
                except subprocess.CalledProcessError as e:
                    logging.error("Error while executing command: %s", e)
                start_hour = int(end_hour)
                start_min = int(end_min)
                start_sec = int(end_sec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--min", type=str, default='10')
    parser.add_argument("--sr", type=str, default='16000')
    parser.add_argument("--out_path", type=str, default="../raw")
    args = parser.parse_args()

    # Validate and normalize input and output paths
    if not os.path.exists(args.path):
        raise FileNotFoundError(f"Input path '{args.path}' does not exist.")

    outdir = os.path.join(shlex.quote(args.path), shlex.quote(args.out_path))
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    cut_video(args, outdir)
