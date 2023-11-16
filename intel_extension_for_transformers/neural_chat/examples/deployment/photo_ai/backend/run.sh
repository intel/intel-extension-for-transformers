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

# Kill the exist and re-run
ps -ef |grep 'photoai' |awk '{print $2}' |xargs kill -9

# KMP
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0

# OMP
export OMP_NUM_THREADS=52
export LD_PRELOAD=${CONDA_PREFIX}/lib/libiomp5.so

# tc malloc
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so

# environment variables
export MYSQL_PASSWORD="root"
export MYSQL_HOST="127.0.0.1"
export MYSQL_DB="ai_photos"
export MYSQL_PORT=3306
export IMAGE_SERVER_IP="198.175.88.26"
export IMAGE2IMAGE_IP="54.87.46.229"
export IMAGE_ROOT_PATH="/home/nfs_images"
export RETRIEVAL_FILE_PATH="/home/tme/photoai_retrieval_docs"
export GOOGLE_API_KEY="AIzaSyD4m9izGcZnv55l27ZvlymdmNsGK7ri_Gg"

nohup numactl -l -C 0-51 python -m photoai 2>&1 &
