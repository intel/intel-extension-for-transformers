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
modes=acc,perf
medium_n=5
if [ $# -eq 0 ]; then
    echo "At least one argument required to specify log_dir!"
    exit 1
fi
log_dir=$1
socket_id=$2
rm -rf $log_dir
mkdir -p $log_dir

medium_n=5

source $script_dir/benchmark.sh --modes=acc,perf --op=sparse_matmul --medium_n=$medium_n --it_per_core=500 \
    --batch="$script_dir/inputs/ci_vnni_input" --socket=${socket_id} |
    tee "$log_dir/vnni.log"
source $script_dir/benchmark.sh --modes=acc,perf --op=sparse_matmul --medium_n=$medium_n --it_per_core=300 \
    --batch="$script_dir/inputs/ci_amx_bf16_x16_input" --socket=${socket_id} |
    tee "$log_dir/amx_bf16_x16.log"
source $script_dir/benchmark.sh --modes=acc,perf --op=layernorm_ba --medium_n=$medium_n --it_per_core=300 \
    --batch="$script_dir/inputs/ci_layernorm_ba_input" --socket=${socket_id} |
    tee "$log_dir/layernorm_ba.log"
source $script_dir/benchmark.sh --modes=acc,perf --op=eltwiseop --medium_n=$medium_n --it_per_core=300 \
    --batch="$script_dir/inputs/ci_eltwiseop_input" --socket=${socket_id} |
    tee "$log_dir/eltwiseop.log"
source $script_dir/benchmark.sh --modes=acc,perf --op=transpose_matmul --medium_n=$medium_n --it_per_core=300 \
    --batch="$script_dir/inputs/ci_matmul_avx512f_8bit_input" --socket=${socket_id} |
    tee "$log_dir/matmul_avx512f_8bit.log"
source $script_dir/benchmark.sh --modes=acc,perf --op=transpose_matmul --medium_n=$medium_n --it_per_core=300 \
    --batch="$script_dir/inputs/ci_matmul_avx512f_p2031_p2013_input" --socket=${socket_id} |
    tee "$log_dir/matmul_avx512f_p2031_p2013.log"
source $script_dir/benchmark.sh --modes=acc,perf --op=transpose_matmul --medium_n=$medium_n --it_per_core=300 \
    --batch="$script_dir/inputs/ci_matmul_vnni_p2031_p2013_input" --socket=${socket_id} |
    tee "$log_dir/matmul_vnni_p2031_p2013.log"
source $script_dir/benchmark.sh --modes=acc,perf --op=transpose_matmul --medium_n=$medium_n --it_per_core=300 \
    --batch="$script_dir/inputs/ci_matmul_vnni_noperm_p2031_p1302_input" --socket=${socket_id} |
    tee "$log_dir/matmul_vnni_noperm_p2031_p1302.log"
source $script_dir/benchmark.sh --modes=acc,perf --op=softmax --medium_n=$medium_n --it_per_core=300 \
    --batch="$script_dir/inputs/ci_softmax_input" --socket=${socket_id} |
    tee "$log_dir/softmax.log"
source $script_dir/benchmark.sh --modes=acc,perf --op=attention --medium_n=$medium_n --it_per_core=300 \
    --batch="$script_dir/inputs/ci_attention_input" --socket=${socket_id} |
    tee "$log_dir/attention.log"
source $script_dir/benchmark.sh --modes=acc,perf --op=transpose_mha --medium_n=$medium_n --it_per_core=300 \
    --batch="$script_dir/inputs/ci_transpose_mha_input" --socket=${socket_id} |
    tee "$log_dir/transpose_mha.log"
source $script_dir/benchmark.sh --modes=acc,perf --op=mha_dense --medium_n=$medium_n --it_per_core=100 \
    --batch="$script_dir/inputs/ci_mha_dense_input" --socket=${socket_id} |
    tee "$log_dir/mha_dense.log"
source $script_dir/benchmark.sh --modes=acc,perf --op=dynamic_quant_matmul --medium_n=$medium_n --it_per_core=300 \
    --batch="$script_dir/inputs/ci_dynamic_quant_matmul_input" --socket=${socket_id} |
    tee "$log_dir/dynamic_quant_matmul.log"
source $script_dir/benchmark.sh --modes=acc --op=dynamic_quant --medium_n=$medium_n --it_per_core=300 \
    --batch="$script_dir/inputs/ci_dynamic_quant_input" --socket=${socket_id} |
    tee "$log_dir/dynamic_quant.log"
source $script_dir/benchmark.sh --modes=acc,perf --op=mha_dense --medium_n=$medium_n --it_per_core=300 \
    --batch="$script_dir/inputs/ci_mha_dense_dynamic_input" --socket=${socket_id} |
    tee "$log_dir/mha_dense_dynamic.log"
source $script_dir/benchmark.sh --modes=acc,perf --op=mha_dense --medium_n=$medium_n --it_per_core=100 \
    --batch="$script_dir/inputs/ci_mha_dense_bf16_input" --socket=${socket_id} |
    tee "$log_dir/mha_dense_bf16.log"
