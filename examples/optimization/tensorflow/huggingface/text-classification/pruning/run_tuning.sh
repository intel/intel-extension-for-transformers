#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {
  tuned_checkpoint=saved_results
  topology="distilbert_base_sst2"
  # topology="bert_base_mrpc_static"
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
      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;
    esac
  done

}

# run_tuning
function run_tuning {
    extra_cmd=''
    batch_size=64
    if [ "${topology}" = "distilbert_base_sst2" ]; then
        TASK_NAME='sst2'
        model_name_or_path=distilbert-base-uncased-finetuned-sst-2-english
    fi

    if [ "${worker}" = "" ]
    then
        python -u ./run_glue.py \
            --model_name_or_path ${model_name_or_path} \
            --task_name ${TASK_NAME} \
            --target_sparsity_ratio 0.1 \
            --prune \
            --do_eval \
            --do_train \
            --per_device_train_batch_size ${batch_size} \
            --per_device_eval_batch_size ${batch_size} \
            --output_dir ${tuned_checkpoint} \
            --overwrite_output_dir \
            --overwrite_cache
    else
        python -u ./run_glue.py \
            --model_name_or_path ${model_name_or_path} \
            --task_name ${TASK_NAME} \
            --target_sparsity_ratio 0.1 \
            --prune \
            --do_eval \
            --do_train \
            --per_device_train_batch_size ${batch_size} \
            --per_device_eval_batch_size ${batch_size} \
            --output_dir ${tuned_checkpoint} \
            --overwrite_output_dir \
            --overwrite_cache \
            --worker "${worker}" \
            --task_index ${task_index} \
            ${extra_cmd}
    fi
}

main "$@"
