#!/bin/bash

script_dir=$(dirname "${BASH_SOURCE[0]}")
log_dir=$1
rm -rf $log_dir
mkdir -p $log_dir

medium_n=1

source $script_dir/benchmark.sh --modes=acc,perf --op=sparse_matmul --medium_n=$medium_n --it_per_core=300 \
    --batch="$script_dir/inputs/ci_vnni_input" |
    tee "$log_dir/vnni.log"
source $script_dir/benchmark.sh --modes=acc,perf --op=layernorm_ba --medium_n=$medium_n --it_per_core=300 \
    --batch="$script_dir/inputs/ci_layernorm_ba_input" |
    tee "$log_dir/layernorm_ba.log"
source $script_dir/benchmark.sh --modes=acc,perf --op=eltwiseop --medium_n=$medium_n --it_per_core=300 \
    --batch="$script_dir/inputs/ci_eltwiseop_input" |
    tee "$log_dir/eltwiseop.log"
