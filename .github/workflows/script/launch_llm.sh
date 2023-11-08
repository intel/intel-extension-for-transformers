#!/bin/bash
set -x
set -eo pipefail

cores_list=(56)
batch_size_list=(1 4)
input_list=(32 512)
output_list=(32 128)
beam_list=(1 4)

function main() {
    conda_env="$1"
    model="$2"
    working_dir="$3"
    log_prefix="$4"
    script="${working_dir}/run_llm.py"
    precision="$5"
    # init params
    if [[ "${model}" == "gpt-j-6b" ]] || [[ "${model}" == "gpt-j-6b-pruned" ]]; then
        model_name="EleutherAI/gpt-j-6B"
    elif [[ "${model}" == "llama-2-7b-chat" ]]; then
        model_name="meta-llama/Llama-2-7b-chat-hf"
    fi

    # init conda
    #. $(dirname ${CONDA_EXE})/../etc/profile.d/conda.sh
    conda activate $conda_env || source activate $conda_env

    # env
    export KMP_BLOCKTIME=1
    export KMP_SETTINGS=1
    export KMP_AFFINITY=granularity=fine,compact,1,0
    export LD_PRELOAD=${CONDA_PREFIX}/lib/libiomp5.so:${CONDA_PREFIX}/lib/libtcmalloc.so
    export GLOG_minloglevel=2

    # launch benchmark
    for cores_per_instance in ${cores_list[@]}; do
        for batch_size in ${batch_size_list[@]}; do
            for input in ${input_list[@]}; do
                [[ "${input}" == "32" ]] && output=32 || output=128
                #sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
                logs_file="${model}-${precision}-${cores_per_instance}-${batch_size}-${input}-${output}.log"
                ir_path="${working_dir}/${precision}_ir"
                python ${WORKING_DIR}/.github/workflows/script/py_task_injection.py --task=get_ITREX_cpu_memory_info --file_name=${script}
                if [[ ${precision} == "fp8" ]]; then
                    export NE_WEIGHT_FP8_4E3M=1
                    ir_path="${working_dir}/bf16_ir"
                    numactl -m 1 -C 56-111 python ${script} --weight_type=fp8_4e3m --input-tokens $input --max-new-tokens $output --batch-size $batch_size --model_path ${ir_path} --model ${model_name} 2>&1 | tee ${WORKING_DIR}/${logs_file} || true
                else
                    numactl -m 1 -C 56-111 python ${script} --input-tokens $input --max-new-tokens $output --batch-size $batch_size --model_path ${ir_path} --model ${model_name} 2>&1 | tee ${WORKING_DIR}/${logs_file} || true
                fi  
                collect_perf_logs_llm ${logs_file} ${precision}
            done
        done
    done

    conda deactivate >/dev/null 2>&1
}

function collect_perf_logs_llm {
    # latency
    log_dir="${WORKING_DIR}/$1"
    latency=($(grep -i 'inference latency:' ${log_dir} | sed -e 's/.*atency://;s/[^0-9.]//g;s/\.$//' | awk '
        BEGIN {
            num = 0;
            sum = 0;
        }{
            num ++;
            sum += $1;
        }END {
            if(num > 0) {
                printf("%d  %.6f", num, sum / num);
            }else {
                printf("0  0");
            }
        }
    '))
    input_tokens=$input
    max_new_tokens=$output
    beam_search=4
    # throughput
    throughput=($(
        echo | awk -v bs=$batch_size -v it=$input -v sec=${latency[1]} -v i=${latency[0]} '{
            if(sec <= 0) {
                print "0";
            }else {
                printf("%.3f", bs * it / sec * i);
            }
        }'
    ) 0)
    # memory usage
    used_memory=$(grep 'memory used total:' ${log_dir} | tail -n 1 | head -n 1 | awk '{print $(NF-1)}')
    # summary
    framework="engine"
    mode_name="latency"
    precision=$2
    link="${log_prefix}/$1"
    printf "${framework},${mode_name},${model},${precision},${batch_size}," | tee -a ${WORKING_DIR}/llm_summary.log
    printf "${input_tokens},${max_new_tokens},${beam_search},${used_memory}," | tee -a ${WORKING_DIR}/llm_summary.log
    printf "${cores_per_instance},${latency[0]},${throughput[0]},${link} ," | tee -a ${WORKING_DIR}/llm_summary.log
    printf "${latency[1]},${first_latency},${avg_latency},${p90_latency},$(hostname)\n" | tee -a ${WORKING_DIR}/llm_summary.log
    set +x
    echo -e "\n\n-------- Summary --------"
    sed -n '1p;$p' ${WORKING_DIR}/llm_summary.log | column -t -s ','
}

main $@ 2>&1 | tee ${WORKING_DIR}/launch.log
