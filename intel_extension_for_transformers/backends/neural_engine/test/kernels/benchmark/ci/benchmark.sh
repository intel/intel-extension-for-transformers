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

# Note: This script can only be run with bash. See: https://stackoverflow.com/a/1221870/21847662

ncores_per_socket=$(lscpu | grep "Core(s) per socket:" | sed -r 's/.+?:\s+(.+)/\1/')
no_numa_support=$(numactl -s | grep "No NUMA support available")
TIMEOUT_DURATION=30 # timeout duration in seconds (for each instance for each run)

function run_multi_inst {
    local ncores_per_inst=$1
    local cmd=$2
    local unified_log=$3

    echo -e ">>> run run_multi_inst $ncores_per_inst $cmd" >>"$unified_log"
    for ((j = 0; $(($j + $ncores_per_inst)) <= $ncores_per_socket; j = $(($j + ${ncores_per_inst})))); do
        local numa_prefix="numactl -m 0 -C $j-$((j + ncores_per_inst - 1)) "
        # Make it works on machines with no numa support
        if [[ -n $no_numa_support ]]; then
            local numa_prefix=""
            export OMP_NUM_THREADS=$ncores_per_inst
        fi
        echo "${numa_prefix}${cmd}" >>$unified_log
        ${numa_prefix}timeout -v $TIMEOUT_DURATION ${cmd} 2>&1 |
            tee -a $unified_log |
            grep "kernel execution time" |
            sed -r "s/^.*?kernel execution time:\s*(.+?)ms,\s*GFLOPS:\s*(.+?)$/\1 \2/" &
    done
    wait
    echo -e "<<< end run_multi_inst $1 $2" >>"$unified_log"
}

function get_medium_result {
    local medium_n=$1
    local ncores_per_inst=$2
    local cmd=$3
    local unified_log=$4

    rm -f tmp_results_file
    for ((j = 0; $j < $medium_n; j = $(($j + 1)))); do
        run_multi_inst "$ncores_per_inst" "$cmd" "$unified_log" |
            awk -v ncores_per_inst=$ncores_per_inst '{
                    sum_time+=$1;
                    sum_gflops+=$2;
                    count+=1;
                } END {
                    if (count!=0) {
                        ave_time=sum_time/count;
                        ave_gflops=sum_gflops/count/ncores_per_inst;
                    };
                    printf("%s,%s\n", ave_time, ave_gflops);
                }' >>tmp_results_file 2>&1
        wait
    done
    cat tmp_results_file |
        sort -t "," -k 2 -r |            # desc sort based on gflops
        sed -n $((($medium_n + 1) / 2))p # get medium
    rm -f tmp_results_file
}

function asset_non_empty {
    local varname=$1
    if [[ ! (-n ${!varname}) ]]; then
        printf "*************************************\n"
        printf "* Error: no $1 specified\n"
        printf "*************************************\n"
        exit 1
    fi
}

export BENCHMARK_NO_REFRESH=1
echo "$@"
if [[ !($WORKSPACE) ]]; then WORKSPACE="."; fi

# default values
modes=perf
medium_n=10
it_per_core=1000

# parse arguments
while [ $# -gt 0 ]; do
    case "$1" in
    --cmake_dir=*)
        cmake_dir="${1#*=}"
        ;;
    --vtune=*)
        vtune_dir="${1#*=}"
        if [[vtune_dir == "1"]]; then
            vtune_dir="/opt/intel/oneapi/vtune/latest/"
        fi
        ;;
    --op=*)
        op="${1#*=}"
        ;;
    --batch=*)
        batch_path="${1#*=}"
        ;;
    --medium_n=*)
        medium_n="${1#*=}"
        ;;
    --modes=*)
        modes="${1#*=}"
        modes=$(echo $modes | tr ',' " ")
        ;;
    --it_per_core=*)
        it_per_core="${1#*=}"
        ;;
    --raw_log=*)
        raw_log="${1#*=}"
        ;;
    *)
        printf "*************************************\n"
        printf "* Error: Invalid argument: \"${1}\" *\n"
        printf "*************************************\n"
        exit 1
        ;;
    esac
    shift
done
asset_non_empty "batch_path"
asset_non_empty "modes"
asset_non_empty "op"
if [[ !($raw_log) ]]; then raw_log="${WORKSPACE}/benchmark_raw_$op-$(basename $batch_path).log"; fi
rm -f $raw_log
dirname $raw_log | xargs mkdir -p
touch $raw_log

# define configs
curr_branch="$(git rev-parse HEAD | sed -r 's/^(.{8}).+/\1/')_$(git rev-parse --abbrev-ref HEAD | sed 's/\//./g')"
echo "curr_branch=$curr_branch"

# Rebuild the project if cmake_dir is defined
if [[ -n $cmake_dir ]]; then
    if [[ -n $vtune_dir ]]; then
        cmake_args="-DNE_WITH_SPARSELIB_VTUNE=True -DCMAKE_VTUNE_HOME=${vtune_dir}"
    fi
    cmake $cmake_dir $cmake_args
    make -j
fi

# For each line in the batch file
while read -r config || [[ -n "${config}" ]]; do
    # set env
    if [[ -n $(echo $config | grep -E '^\$') ]]; then
        env_chnage=$(echo $config | sed -e "s/^\$\s*//")
        echo "export $env_chnage" >>$raw_log
        export $env_chnage
        continue
    fi
    # print line
    if [[ -n $(echo $config | grep -E '^###') ]]; then
        echo $config | sed -e "s/###\s*//" | xargs -d"\n" printf "%s %s\n" ">>>>>>>###"
    fi
    # skip comment line and empty line
    if [[ -n $(echo $config | grep -E '^#') ]] || [[ -z $config ]]; then continue; fi

    printf "%s %s\n" ">>>>>>>>>>" "$config"
    ncores_per_instance=$(echo $config | cut -d' ' -f1)
    spmm_params=$(echo $config | cut -d' ' -f2-)

    export BENCHMARK_ITER=$(($it_per_core * $ncores_per_instance))
    for mode in $modes; do
        cmd="./benchmark $mode $op $spmm_params"

        if [[ $mode == "acc" ]]; then
            echo ">>> run $cmd" >>$raw_log
            acc_result=$(
                $cmd 2>&1 | tee -a $raw_log
                exit ${PIPESTATUS[0]}
            )
            cmd_exit_code=$?
            echo "<<< end $cmd" >>$raw_log
            if (test $cmd_exit_code -eq 0) && (test -n "$(echo $acc_result | grep "result correct")"); then
                echo "result correct"
            else
                echo "benchmark failed"
            fi
        else
            get_medium_result "$medium_n" "$ncores_per_instance" "$cmd" "$raw_log"
        fi
    done
    echo

done <"$batch_path"
wait
