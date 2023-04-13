#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  topology="bert_base_mrpc_static"
  iters=100
  batch_size=1
  tuned_checkpoint=saved_results
  cache_dir="cache"
  for var in "$@"
  do
    case $var in
      --topology=*)
          topology=$(echo $var |cut -f2 -d=)
      ;;
      --dataset_location=*)
          dataset_location=$(echo $var |cut -f2 -d=)
      ;;
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --mode=*)
          mode=$(echo $var |cut -f2 -d=)
      ;;
      --batch_size=*)
          batch_size=$(echo $var |cut -f2 -d=)
      ;;
      --iters=*)
          iters=$(echo ${var} |cut -f2 -d=)
      ;;
      --int8=*)
          int8=$(echo ${var} |cut -f2 -d=)
      ;;
      --config=*)
          tuned_checkpoint=$(echo $var |cut -f2 -d=)
      ;;
      --worker=*)
          worker=$(echo $var |cut -f2 -d=)
      ;;
      --task_index=*)
          task_index=$(echo $var |cut -f2 -d=)
      ;;
      --cache_dir=*)
          cache_dir=$(echo $var |cut -f2 -d=)
      ;;
      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;
    esac
  done

}


# run_benchmark
function run_benchmark {
    extra_cmd=''
    MAX_SEQ_LENGTH=128

    if [[ ${mode} == "accuracy" ]]; then
        mode_cmd=" --accuracy_only"
    elif [[ ${mode} == "benchmark" ]]; then
        mode_cmd=" --benchmark "
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi

    if [ "${topology}" = "bert_base_mrpc_static" ]; then
        TASK_NAME="mrpc"
        model_name_or_path="bert-base-cased-finetuned-mrpc"
    elif [ "${topology}" = "legalbert_mrpc" ]; then
        TASK_NAME="mrpc"
        model_name_or_path="nlpaueb/legal-bert-small-uncased"
    elif [ "${topology}" = "xlnet_mrpc" ]; then
        TASK_NAME="mrpc"
        model_name_or_path="xlnet-base-cased"
    elif [ "${topology}" = "albert_large_mrpc" ]; then
        TASK_NAME="mrpc"
        model_name_or_path="albert-large-v2"
        # add following parameters for quicker debugging
        extra_cmd=$extra_cmd" --max_eval_samples 48"
    fi

    if [[ ${int8} == "true" ]]; then
        extra_cmd=$extra_cmd" --int8"
    fi
    echo $extra_cmd

    if [ "${worker}" = "" ]
    then
        python -u ../run_glue.py \
            --model_name_or_path ${model_name_or_path} \
            --task_name ${TASK_NAME} \
            --do_eval \
            --max_seq_length ${MAX_SEQ_LENGTH} \
            --per_device_eval_batch_size ${batch_size} \
            --output_dir ${tuned_checkpoint} \
            --overwrite_output_dir \
            --cache_dir ${cache_dir} \
            --no_cuda \
            ${mode_cmd} \
            ${extra_cmd}
    else
        python -u ../run_glue.py \
            --model_name_or_path ${model_name_or_path} \
            --task_name ${TASK_NAME} \
            --do_eval \
            --max_seq_length ${MAX_SEQ_LENGTH} \
            --per_device_eval_batch_size ${batch_size} \
            --output_dir ${tuned_checkpoint} \
            --overwrite_output_dir \
            --cache_dir ${cache_dir} \
            --no_cuda \
            --worker "${worker}" \
            --task_index ${task_index} \
            ${mode_cmd} \
            ${extra_cmd}
    fi
}

main "$@"
