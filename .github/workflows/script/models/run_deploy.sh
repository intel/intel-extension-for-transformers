#!/bin/bash
set -eo pipefail
source /intel-extension-for-transformers/.github/workflows/script/change_color.sh

# get parameters
PATTERN='[-a-zA-Z0-9_]*='
PERF_STABLE_CHECK=true

for i in "$@"; do
    case $i in
        --framework=*)
            framework=`echo $i | sed "s/${PATTERN}//"`;;
        --model=*)
            model=`echo $i | sed "s/${PATTERN}//"`;;
        --mode=*)
            mode=`echo $i | sed "s/${PATTERN}//"`;;
        --log_dir=*)
            log_dir=`echo $i | sed "s/${PATTERN}//"`;;
        --precision=*)
            precision=`echo $i | sed "s/${PATTERN}//"`;;
        --PERF_STABLE_CHECK=*)
            PERF_STABLE_CHECK=`echo $i | sed "s/${PATTERN}//"`;;
        *)
            echo "Parameter $i not recognized."; exit 1;;
    esac
done

log_dir="/intel-extension-for-transformers/${framework}_${model}"
mkdir -p ${log_dir}
$BOLD_YELLOW && echo "-------- run_benchmark_common --------" && $RESET

main() {
    ## prepare
    prepare

    ## run accuracy
    if [[ $(echo "${mode}" | grep "accuracy") ]]; then
        run_benchmark "accuracy" 8
    fi

    ## run accuracy
    if [[ $(echo "${mode}" | grep "latency") ]]; then
        run_benchmark "latency" 1
    fi

    # run performance
    if [[ $(echo "${mode}" | grep "performance") ]] && [[ ${PERF_STABLE_CHECK} == "false" ]]; then
        run_benchmark "throughput" 1
    elif [[ $(echo "${mode}" | grep "performance") ]]; then
        max_loop=3
        gap=(0.05 0.05 0.1)
        for ((iter = 0; iter < ${max_loop}; iter++)); do
            run_benchmark "throughput" 1
            {
                check_perf_gap ${gap[${iter}]}
                exit_code=$?
            } || true

            if [ ${exit_code} -ne 0 ]; then
                $BOLD_RED && echo "FAILED with performance gap!!" && $RESET
            else
                $BOLD_GREEN && echo "SUCCEED!!" && $RESET
                break
            fi
        done
    fi

    ## run inferencer
    if [[ $(echo "${mode}" | grep "performance") ]] && [[ ${framework} == "engine" ]] && [[ ${precision} != "dynamic_int8" ]]; then
        run_inferencer
    fi

     exit ${exit_code}
}

function check_perf_gap() {
    python -u /intel-extension-for-transformers/.github/workflows/script/models/collect_model_log.py \
        --framework=${framework} \
        --model=${model} \
        --logs_dir="${log_dir}" \
        --output_dir="${log_dir}" \
        --build_id='0' \
        --stage="${precision}_benchmark" \
        --gap=$1
}

function prepare() {
    if [[ ${model} == "bert_mini_sparse" ]]; then
        working_dir="/intel-extension-for-transformers/examples/huggingface/pytorch/text-classification/deployment/sparse/bert_mini"
    elif [[ ${model} == "distilbert_base_squad_ipex" ]]; then
        working_dir="/intel-extension-for-transformers/examples/huggingface/pytorch/question-answering/deployment/squad/ipex/distilbert_base_uncased"
    fi
    cd ${working_dir}
    echo "Working in ${working_dir}"
    echo -e "\nInstalling model requirements..."
    if [ -f "requirements.txt" ]; then
        python -m pip install -r requirements.txt 
        pip list
    else
        echo "Not found requirements.txt file."
    fi
}

function run_benchmark() {
    local input_mode=$1
    local batch_size=$2
    if [[ ${model} == "bert_mini_sparse" ]]; then
        benchmark_cmd="bash run_bert_mini.sh --mode=${input_mode} --precision=${precision} --model=Intel/bert-mini-sst2-distilled-sparse-90-1X4-block --dataset=sst2 --output=${log_dir} --log_name=engine-bert_mini_sparse-${precision}-linux-icx --batch_size=${batch_size}"
    elif [[ ${model} == "distilbert_base_squad_ipex" ]]; then
        benchmark_cmd="bash run_distilbert.sh --mode=${input_mode} --precision=${precision} --model=distilbert-base-uncased-distilled-squad --dataset=squad --output=${log_dir} --log_name=ipex-distilbert_base_squad_ipex-${precision}-linux-icx --batch_size=${batch_size}"
    fi
    cd ${working_dir}
    overall_log="${log_dir}/${framework}-${model}-${precision}-${input_mode}.log"
    ${benchmark_cmd} 2>&1 | tee ${overall_log}
}

function run_inferencer() {
    if [[ ${model} == "bert_mini_sparse" ]]; then
        ir_path="${working_dir}/sparse_${precision}_ir"
        inference_cmd="bash -x /intel-extension-for-transformers/.github/workflows/script/launch_benchmark.sh "${model}" "${ir_path}" "8" "1" "${precision}" "${working_dir}" "0""
    fi
    overall_log="${log_dir}/inferencer_${model}_${precision}.log"
    eval ${inference_cmd} 2>&1 | tee $overall_log
}

main
