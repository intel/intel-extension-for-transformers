#!/bin/bash
#===============================================================================
# Copyright (c) 2023 Intel Corporation
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
set -x
quant_nthr=48
cores_list=(32 48 56)
#cores_list=(48)
batch_size_list=(1)
#input_list=(10 32 1024 2012)
input_list=(32 1024 2012)
output_list=(32)
beam_list=(1)
# input_list=(32 512)
# output_list=(32 128 512)
extra_precision_list=("q4_j_i8_g128" "q4_j_i8_g32" "q4_0") # precisions to be tested for most of supported models

ppl_dataset_list=("/tf_dataset2/datasets/nlp_toolkit/wikitext-2-raw-v1-data-test")
ppl_nctx_list=() # no ppl test by defalut
# ppl_nctx_list=(256 1024 2048)
drop_caches=false
ppl_fp32_test=false
ppl_mf16_test=false

# parse named arguments
while [ $# -gt 0 ]; do
    case "$1" in
    --cores_list=*)
        IFS=', ' read -r -a cores_list <<<"${1#*=}"
        ;;
    --batch_size_list=*)
        IFS=', ' read -r -a batch_size_list <<<"${1#*=}"
        ;;
    --input_list=*)
        # list of input length
        IFS=', ' read -r -a input_list <<<"${1#*=}"
        ;;
    --output_list=*)
        IFS=', ' read -r -a output_list <<<"${1#*=}"
        ;;
    --beam_list=*)
        IFS=', ' read -r -a beam_list <<<"${1#*=}"
        ;;
    --ppl_dataset_list=*)
        IFS=', ' read -r -a ppl_dataset_list <<<"${1#*=}"
        ;;
    --extra_precision_list=*)
        # tested precisions = unique(extra_precision_list + model-specific precision)
        IFS=', ' read -r -a extra_precision_list <<<"${1#*=}"
        ;;
    --drop_caches)
        # be careful to turn this on; it requires sudo
        drop_caches=true
        ;;
    --ppl_nctx_list=*)
        # set to non-empty to enable ppl test
        IFS=', ' read -r -a ppl_nctx_list <<<"${1#*=}"
        ;;
    --ppl_fp32_test)
        ppl_fp32_test=true
        ;;
    --ppl_mf16_test)
        # be careful to turn this on; it will double the workload
        ppl_mf16_test=true
        ;;
    --)
        shift
        break
        ;;
    *)
        break
        ;;
    esac
    shift
done

declare -p cores_list
declare -p batch_size_list
declare -p input_list
declare -p output_list
declare -p beam_list
declare -p extra_precision_list

declare -p ppl_dataset_list
declare -p ppl_nctx_list
declare -p ppl_fp32_test
declare -p ppl_mf16_test

function ppl_eval() {
    local task_name="$1"
    local n_cores="$2"
    local input_model="$3"
    local quantized_weight_path="$4"
    local memory_dtype_list=('auto')
    if [[ "$ppl_mf16_test" = true ]]; then
        memory_dtype_list+=('f16')
    fi

    echo "=======  PPL Evaluation Start  ======="
    for memory_dtype in ${memory_dtype_list[@]}; do
        for ppl_dataset in ${ppl_dataset_list[@]}; do
            for ppl_nctx in ${ppl_nctx_list[@]}; do
                local ppl_task_name="$task_name-ppl-$(basename -- "$ppl_dataset")-nctx$ppl_nctx-M$memory_dtype"
                echo "***** PPL: $ppl_task_name *****"
                OMP_NUM_THREADS=$(($n_cores * 1)) numactl -m 0 -C 0-$(($n_cores * 1 - 1)) \
                    python scripts/perplexity.py --model_name "$input_model" --dataset_name "$ppl_dataset" --quantized_weight_path "$quantized_weight_path" --ctx_size $ppl_nctx --n_threads $n_cores --memory_dtype $memory_dtype 2>&1 |
                    tee "$WORKSPACE/$ppl_task_name.log"
                mv out/ppl.png "$WORKSPACE/$ppl_task_name.png"
                mv out/ppl_data.json "$WORKSPACE/$ppl_task_name.json"
            done
        done
    done
    echo "=======  PPL Evaluation End  ======="
}

function main() {
    conda_env="$1"
    model="$2"
    working_dir="$3"       # The "graph" directory with CMakeFile.txt
    export log_prefix="$4" # The log url prefix
    compiler_version="$5"
    PROMPTS_PATH="$script_dir/cpp_graph_prompts.json"

    # init params
    precision_list=()
    requirements_file="requirements.txt"  # some models need extra constraints
    if [[ "${model}" == "llama-2-7b-chat" ]]; then
        quant_script="./build/bin/quant_llama"
        infer_cmd="./build/bin/run_llama"
        input_model="/tf_dataset2/models/nlp_toolkit/llama-2-7b-chat/Llama-2-7b-chat-hf"
        precision_list+=( # Our "main model"
            "q4_j1_i8_g128" "q4_j_f32_g128" "q4_1" "q8_0"
            "q8e4m3_j_f32_g128" "q8e4m3_j_f32_g128_fp8" "q8e5m2_j_f32_g128" "q8e5m2_j_f32_g128_fp8"
            "q4e2m1_j_f32_g128" "nf4_j_f32_g128"
        )
    elif [[ "${model}" == "gptj-6b" ]]; then
        quant_script="./build/bin/quant_gptj"
        infer_cmd="./build/bin/run_gptj"
        model_name="EleutherAI/gpt-j-6b"
        input_model="/tf_dataset2/models/pytorch/gpt-j-6B" # model focused on mlperf
        precision_list+=("q4_j1_i8_g128" "q4_j1_bf16_pc")
    elif [[ "${model}" == "gpt-neox-20b" ]]; then
        quant_script="./build/bin/quant_gptneox"
        infer_cmd="./build/bin/run_gptneox"
        model_name="EleutherAI/gpt-neox-20b"
        input_model="/tf_dataset2/models/nlp_toolkit/gpt-neox-20b"
    elif [[ "${model}" == "mpt-7b" ]]; then
        quant_script="./build/bin/quant_mpt"
        infer_cmd="./build/bin/run_mpt"
        model_name="mosaicml/mpt-7b"
        input_model="/tf_dataset2/models/nlp_toolkit/mpt-7b"
    elif [[ "${model}" == "falcon-7b" ]]; then
        quant_script="./build/bin/quant_falcon"
        infer_cmd="./build/bin/run_falcon"
        model_name="tiiuae/falcon-7b"
        input_model="/tf_dataset2/models/nlp_toolkit/falcon-7b"
    elif [[ "${model}" == "starcoder-3b" ]]; then
        quant_script="./build/bin/quant_starcoder"
        infer_cmd="./build/bin/run_starcoder"
        model_name="bigcode/starcoder"
        input_model="/tf_dataset2/models/pytorch/starcode_3b"
    elif [[ "${model}" == "bloom-7b" ]]; then
        quant_script="./build/bin/quant_bloom"
        infer_cmd="./build/bin/run_bloom"
        model_name="bigscience/bloom-7b1"
        input_model="/tf_dataset2/models/pytorch/bloom-7b1"
    elif [[ "${model}" == "opt-1.3b" ]]; then
        quant_script="./build/bin/quant_opt"
        infer_cmd="./build/bin/run_opt"
        model_name="facebook/opt-1.3b"
        input_model="/tf_dataset2/models/pytorch/opt-1.3b"
    elif [[ "${model}" == "dolly-v2-3b" ]]; then
        quant_script="./build/bin/quant_dolly"
        infer_cmd="./build/bin/run_dolly"
        model_name="databricks/dolly-v2-3b"
        input_model="/tf_dataset2/models/pytorch/dolly_v2_3b"
    elif [[ "${model}" == "chatglm2" ]]; then
        quant_script="./build/bin/quant_chatglm2"
        infer_cmd="./build/bin/run_chatglm2"
        model_name="THUDM/chatglm2-6b"
        input_model="/tf_dataset2/models/pytorch/chatglm2-6b"
    elif [[ "${model}" == "chatglm-6b" ]]; then
        quant_script="./build/bin/quant_chatglm"
        infer_cmd="python ./scripts/inference.py"
        model_name="THUDM/chatglm-6b"
        input_model="/tf_dataset2/models/pytorch/chatglm-6b"
        extension=" --model_name chatglm --tokenizer $input_model"
        requirements_file="scripts/requirements/chatglm-6b.txt"
    elif [[ "${model}" == "baichuan2-13b" ]]; then
        quant_script="./build/bin/quant_baichuan"
        infer_cmd="python ./scripts/inference.py"
        model_name="baichuan-inc/Baichuan2-13B-Chat"
        input_model="/tf_dataset2/models/pytorch/Baichuan2-13B-Chat"
        extension=" --model_name baichuan --tokenizer $input_model"
    elif [[ "${model}" == "baichuan-13b" ]]; then
        quant_script="./build/bin/quant_baichuan"
        infer_cmd="python ./scripts/inference.py"
        model_name="baichuan-inc/Baichuan-13B-Chat"
        input_model="/tf_dataset2/models/pytorch/Baichuan-13B-Chat"
        extension=" --model_name baichuan --tokenizer $input_model"
    elif [[ "${model}" == "mistral-7b" ]]; then
        quant_script="./build/bin/quant_mistral"
        infer_cmd="./build/bin/run_mistral"
        model_name="mistralai/Mistral-7B-v0.1"
        input_model="/tf_dataset2/models/pytorch/Mistral-7B-v0.1"
        requirements_file="scripts/requirements/mistral.txt"
    elif [[ "${model}" == "qwen-7b" ]]; then
        quant_script="./build/bin/quant_qwen"
        infer_cmd="./build/bin/run_qwen"
        model_name="Qwen/Qwen-7B-Chat"
        input_model="/tf_dataset2/models/nlp_toolkit/Qwen-7B-Chat"
    else
        echo "Error: Unexpedted model: $model" 1>&2
        exit 1
    fi

    if [[ $(lscpu | grep i9-12900 | wc -l) != 0 ]]; then
        cores_list=(16)
        quant_nthr=12
        precision_list+=("q4_j_f32_g128")
    fi

    # add additional precisions
    declare -A precisions_seen
    for p in "${precision_list[@]}"; do
        precisions_seen[$p]=x
    done
    for p in "${extra_precision_list[@]}"; do
        [[ ${precisions_seen[$p]} ]] && continue
        precision_list+=("$p")
        precisions_seen[$p]=x
    done

    # init conda
    #. $(dirname ${CONDA_EXE})/../etc/profile.d/conda.sh
    conda activate $conda_env || source activate $conda_env
    pip install cmake ninja psutil
    if [[ "${compiler_version}" != "12.1.0" ]]; then
        conda install --update-deps -c conda-forge gxx==${compiler_version} gcc==${compiler_version} gxx_linux-64==${compiler_version} libstdcxx-ng sysroot_linux-64 -y
    fi

    # setup conda env for LLM

    # get cpu info
    # sockets=$(lscpu |grep 'Socket(s):' |sed 's/.*://;s/ //g')
    # cores_per_instance=$(lscpu |grep 'Core(s) per socket:' |sed 's/.*://;s/ //g')

    # compile binary
    cd ${working_dir}
    git submodule update --init --recursive -- ./application/third_party
    mkdir build
    cd build
    cmake .. -G Ninja
    ninja
    cd ..

    ## prepare example requiement
    pip install -r "$requirements_file"
    if [[ "${model}" == "baichuan"* ]] || [[ "${model}" == "mistral-7b" ]]; then
        pip install --force-reinstall transformers==4.33.1
    fi
    echo "=======  Convert Start  ======="
    ## prepare fp32 bin
    python "$working_dir/scripts/convert.py" --outtype f32 --outfile ${working_dir}/${model}-fp32.bin ${input_model}
    echo "=======  Convert End  ======="

    # launch benchmark
    for cores_per_instance_idx in "${!cores_list[@]}"; do
        cores_per_instance=${cores_list[cores_per_instance_idx]}
        for batch_size_idx in "${!batch_size_list[@]}"; do
            batch_size=${batch_size_list[batch_size_idx]}
            for input_idx in "${!input_list[@]}"; do
                input=${input_list[input_idx]}
                for precision_idx in "${!precision_list[@]}"; do
                    precision=${precision_list[precision_idx]}
                    # [[ "${input}" == "32" ]] && output=32 ||
                    if [[ "${input}" == "10" ]]; then output=1024; else output=32; fi
                    if [[ "${model}" == "chatglm2" || "${model}" == "chatglm-6b" || "${model}" == "baichuan-13b" || "${model}" == "baichuan2-13b" ]]; then
                        output=32
                    fi
                    prompt=$(python -c "import sys, json; i = json.load(sys.stdin)['$input']; print(i['prompts'][i['map'].get('$model', 'default')])" <$PROMPTS_PATH)

                    if [[ -z $prompt ]]; then
                        echo "Error: Unexpedted input: $input" 1>&2
                        continue
                    fi
                    ctx=$(($output + $input + 10))
                    if [[ "$drop_caches" = true ]]; then
                        sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
                    fi
                    task_name="${model}-${precision}-${cores_per_instance}-${batch_size}-${input}-${output}"
                    logs_file="${task_name}.log"
                    ## prepare model.bin
                    if [[ ! -f ${working_dir}/${model}-${precision}.bin ]]; then
                        echo "=======  Quantization Start  ======="
                        local quant_script_prologue="'$quant_script' --model_file '$working_dir/$model-fp32.bin' --out_file '$working_dir/$model-$precision.bin' --nthread $quant_nthr" # in case there are space in the path
                        if [[ ${precision} == "q4_j_i8_g128" ]]; then
                            eval "$quant_script_prologue --weight_dtype int4 --group_size 128 --compute_dtype int8 --scale_dtype fp32 --alg sym"
                        # deprecated since bfloat16 scale not mature
                        # elif [[ ${precision} == "q4_j_i8_g32_bf16" ]]; then
                        #     eval "$quant_script_prologue --weight_dtype int4 --group_size 32 --compute_dtype int8 --scale_dtype bf16 --alg sym"
                        elif [[ ${precision} == "q4_j_i8_g32" ]]; then
                            eval "$quant_script_prologue --weight_dtype int4 --group_size 32 --compute_dtype int8 --scale_dtype fp32 --alg sym"
                        elif [[ ${precision} == "q4_j_f32_g128" ]]; then
                            eval "$quant_script_prologue --weight_dtype int4 --group_size 128 --compute_dtype fp32 --scale_dtype fp32 --alg sym"
                        elif [[ ${precision} == "q4_j1_i8_g128" ]]; then
                            eval "$quant_script_prologue --weight_dtype int4 --group_size 128 --compute_dtype int8 --scale_dtype fp32 --alg asym"
                        elif [[ ${precision} == "q4_j1_bf16_pc" ]]; then
                            eval "$quant_script_prologue --weight_dtype int4 --group_size -1 --compute_dtype bf16 --scale_dtype fp32 --alg asym"
                        elif [[ ${precision} == "q4_0" ]]; then
                            eval "$quant_script_prologue --weight_dtype int4 --group_size 32 --compute_dtype int8 --alg sym --use_ggml"
                        elif [[ ${precision} == "q4_1" ]]; then
                            eval "$quant_script_prologue --weight_dtype int4 --group_size 32 --compute_dtype int8 --alg asym --use_ggml"
                        elif [[ ${precision} == "q8_0" ]]; then
                            eval "$quant_script_prologue --weight_dtype int8 --group_size 32 --compute_dtype int8 --alg sym --use_ggml"
                        elif [[ ${precision} == "q8e4m3_j_f32_g128" ]]; then
                            eval "$quant_script_prologue --weight_dtype fp8 --group_size 128 --scale_dtype fp32 --compute_dtype fp32 --alg sym"
                        elif [[ ${precision} == "q8e4m3_j_f32_g128_fp8" ]]; then
                            eval "$quant_script_prologue --weight_dtype fp8 --group_size 128 --scale_dtype fp8 --compute_dtype fp32 --alg sym"
                        elif [[ ${precision} == "q4e2m1_j_f32_g128" ]]; then
                            eval "$quant_script_prologue --weight_dtype fp4 --group_size 128 --scale_dtype fp32 --compute_dtype fp32 --alg sym"
                        elif [[ ${precision} == "q8e5m2_j_f32_g128" ]]; then
                            eval "$quant_script_prologue --weight_dtype fp8_e5m2 --group_size 128 --scale_dtype fp32 --compute_dtype fp32 --alg sym"
                        elif [[ ${precision} == "q8e5m2_j_f32_g128_fp8" ]]; then
                            eval "$quant_script_prologue --weight_dtype fp8_e5m2 --group_size 128 --scale_dtype fp8 --compute_dtype fp32 --alg sym"
                        elif [[ ${precision} == "nf4_j_f32_g128" ]]; then
                            eval "$quant_script_prologue --weight_dtype nf4 --group_size 128 --scale_dtype fp32 --compute_dtype fp32 --alg sym"
                        else
                            echo "Error: Unexpedted precision: $precision" 1>&2
                            continue
                        fi
                        echo "=======  Quantization End  ======="
                    fi
                    ## run inference
                    export LANG=en_US.UTF-8
                    export LC_ALL=en_US.UTF-8
                    echo "=======  Inference Start  ======="
                    OMP_NUM_THREADS=$cores_per_instance numactl -m 0 -C 0-$(($cores_per_instance - 1)) \
                        $infer_cmd --seed 1234 -t $cores_per_instance -b ${ctx} -c ${ctx} -n ${output} -m ${model}-${precision}.bin $extension -p "$prompt" 2>&1 | tee ${WORKSPACE}/${logs_file} || true &
                    minitor

                    echo "=======  Inference End  ======="
                    python $script_dir/calculate_percentiles.py ${WORKSPACE}/${logs_file} ${model} ${precision} ${cores_per_instance} ${batch_size} ${input} ${output}

                    if [[ "$cores_per_instance" == "${cores_list[@]: -1:1}" ]] &&
                        [[ "$batch_size_idx" == "0" ]] &&
                        [[ "$input_idx" == "0" ]] &&
                        [[ "${#ppl_nctx_list[@]}" != "0" ]]; then
                        ppl_eval "$task_name" "$cores_per_instance" "$input_model" "$model-$precision.bin"
                    fi
                done
            done
        done
    done
    if [[ "${#ppl_nctx_list[@]}" != "0" ]] && [[ "$ppl_fp32_test" = true ]]; then
        cores_per_instance="${cores_list[@]: -1:1}"
        task_name="${model}-fp32-${cores_per_instance}-${batch_size_list[@]:0:1}-${input_list[@]:0:1}-${output}"
        ppl_eval "$task_name" "$cores_per_instance" "$input_model" "$model-fp32.bin"
    fi
    conda deactivate >/dev/null 2>&1
}

function minitor() {
    sleep 1
    # try first time
    if [ $(ps -ef | grep "$infer_cmd" | wc -l) -lt 2 ]; then
        #python $script_dir/calculate_percentiles.py ${logs_file} ${model} ${precision} ${cores_per_instance} ${batch_size} ${input} ${output}
        sleep 1
    fi
    # keep monitoring
    echo "======  Monitor Start ======="
    while true; do
        if [ $(ps -ef | grep "$infer_cmd" | wc -l) -lt 2 ]; then
            #python $script_dir/calculate_percentiles.py ${logs_file} ${model} ${precision} ${cores_per_instance} ${batch_size} ${input} ${output}
            sleep 3
            break
        fi
        echo "$(date +%s), $(numastat -p $(ps -ef | grep "$infer_cmd" | grep -v grep | awk '{printf("%s  ", $2)}'))" >>${WORKSPACE}/memory.txt 2>&1
    done
    echo "======  Monitor End ======="
}
function get_data() {
    python $script_dir/calculate_percentiles.py ${logs_file} ${model} ${precision} ${cores_per_instance} ${batch_size} ${input} ${output}
}
main $@ 2>&1 | tee ${WORKSPACE}/launch.log
