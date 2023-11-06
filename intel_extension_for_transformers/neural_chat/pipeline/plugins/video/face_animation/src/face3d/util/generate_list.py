
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

"""This script is to generate training list files for Deep3DFaceRecon_pytorch
"""

import os

# save path to training data
def write_list(lms_list, imgs_list, msks_list, mode='train',save_folder='datalist', save_name=''):
    save_path = os.path.join(save_folder, mode)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, save_name + 'landmarks.txt'), 'w') as fd:
        fd.writelines([i + '\n' for i in lms_list])

    with open(os.path.join(save_path, save_name + 'images.txt'), 'w') as fd:
        fd.writelines([i + '\n' for i in imgs_list])

    with open(os.path.join(save_path, save_name + 'masks.txt'), 'w') as fd:
        fd.writelines([i + '\n' for i in msks_list])   

# check if the path is valid
def check_list(rlms_list, rimgs_list, rmsks_list):
    lms_list, imgs_list, msks_list = [], [], []
    for i in range(len(rlms_list)):
        flag = 'false'
        lm_path = rlms_list[i]
        im_path = rimgs_list[i]
        msk_path = rmsks_list[i]
        if os.path.isfile(lm_path) and os.path.isfile(im_path) and os.path.isfile(msk_path):
            flag = 'true'
            lms_list.append(rlms_list[i])
            imgs_list.append(rimgs_list[i])
            msks_list.append(rmsks_list[i])
        print(i, rlms_list[i], flag)
    return lms_list, imgs_list, msks_list
