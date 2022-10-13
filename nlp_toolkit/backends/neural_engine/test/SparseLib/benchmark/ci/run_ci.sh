#!/bin/bash
#===============================================================================
# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

script_dir=$(dirname "${BASH_SOURCE[0]}")
log_dir=$1
rm -rf $log_dir
mkdir -p $log_dir

medium_n=5

source $script_dir/benchmark.sh --modes=acc,perf --op=sparse_matmul --medium_n=$medium_n --it_per_core=200 \
    --batch="$script_dir/inputs/ci_vnni_input" |
    tee "$log_dir/vnni.log"
source $script_dir/benchmark.sh --modes=acc,perf --op=sparse_matmul --medium_n=$medium_n --it_per_core=300 \
    --batch="$script_dir/inputs/ci_amx_bf16_x16_input" |
    tee "$log_dir/amx_bf16_x16.log"
source $script_dir/benchmark.sh --modes=acc,perf --op=layernorm_ba --medium_n=$medium_n --it_per_core=300 \
    --batch="$script_dir/inputs/ci_layernorm_ba_input" |
    tee "$log_dir/layernorm_ba.log"
source $script_dir/benchmark.sh --modes=acc,perf --op=eltwiseop --medium_n=$medium_n --it_per_core=300 \
    --batch="$script_dir/inputs/ci_eltwiseop_input" |
    tee "$log_dir/eltwiseop.log"
source $script_dir/benchmark.sh --modes=acc,perf --op=transpose_matmul --medium_n=$medium_n --it_per_core=300 \
    --batch="$script_dir/inputs/ci_matmul_avx512f_p2031_p2013_input" |
    tee "$log_dir/matmul_avx512f_p2031_p2013_input.log"
