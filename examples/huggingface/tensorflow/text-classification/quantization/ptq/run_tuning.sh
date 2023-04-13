#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {
  topology="bert_base_mrpc_static"
  tuned_checkpoint="saved_results"
  extra_cmd=""
  batch_size=8
  MAX_SEQ_LENGTH=128
  model_type="bert"
  approach="PostTrainingStatic"
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
      --output_model=*)
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

# run_tuning
function run_tuning {
    batch_size=64
    if [ "${topology}" = "bert_base_mrpc_static" ]; then
        TASK_NAME="mrpc"
        model_name_or_path="bert-base-cased-finetuned-mrpc"
        approach="PostTrainingStatic"
    elif [ "${topology}" = "legalbert_mrpc" ]; then
        TASK_NAME="mrpc"
        model_name_or_path="nlpaueb/legal-bert-small-uncased"
        approach="PostTrainingStatic"
        extra_cmd=$extra_cmd" --perf_tol 0.1"
    elif [ "${topology}" = "xlnet_mrpc" ]; then
        TASK_NAME="mrpc"
        model_name_or_path="xlnet-base-cased"
        approach="PostTrainingStatic"
    elif [ "${topology}" = "albert_large_mrpc" ]; then
        TASK_NAME="mrpc"
        model_name_or_path="albert-large-v2"
        approach="PostTrainingStatic"
        extra_cmd=$extra_cmd" --perf_tol 0.05"
    fi
    
    if [ "${worker}" = "" ]
    then
        python -u ../run_glue.py \
            --model_name_or_path ${model_name_or_path} \
            --task_name ${TASK_NAME} \
            --do_eval \
            --max_seq_length ${MAX_SEQ_LENGTH} \
            --per_device_train_batch_size ${batch_size} \
            --per_device_eval_batch_size ${batch_size} \
            --output_dir ${tuned_checkpoint} \
            --no_cuda \
            --overwrite_output_dir \
            --cache_dir ${cache_dir} \
            --quantization_approach ${approach} \
            --do_train \
            --tune \
            ${extra_cmd}
    else
        python -u ../run_glue.py \
            --model_name_or_path ${model_name_or_path} \
            --task_name ${TASK_NAME} \
            --do_eval \
            --max_seq_length ${MAX_SEQ_LENGTH} \
            --per_device_train_batch_size ${batch_size} \
            --per_device_eval_batch_size ${batch_size} \
            --output_dir ${tuned_checkpoint} \
            --no_cuda \
            --overwrite_output_dir \
            --cache_dir ${cache_dir} \
            --quantization_approach ${approach} \
            --do_train \
            --tune \
            --worker "${worker}" \
            --task_index ${task_index} \
            ${extra_cmd}
    fi
}

main "$@"
