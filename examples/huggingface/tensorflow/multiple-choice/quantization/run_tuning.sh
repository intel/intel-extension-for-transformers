#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {
  topology="distilbert"
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
    if [ "${topology}" = "distilbert_swag" ]; then
        script="run_swag.py"
        model_name_or_path="Rocketknight1/bert-base-uncased-finetuned-swag"
        approach="PostTrainingStatic"
        # add following parameters for quicker debugging
        extra_cmd=$extra_cmd" --max_train_samples 512 --max_eval_samples 1024 --perf_tol 0.035"
    fi
    
    if [ "${worker}" = "" ]
    then
        python -u ${script} \
            --model_name_or_path ${model_name_or_path} \
            --output_dir ${tuned_checkpoint} \
            --quantization_approach ${approach} \
            --do_train \
            --tune \
            --overwrite_output_dir \
            --cache_dir ${cache_dir} \
            ${extra_cmd}
    else
        python -u ${script} \
            --model_name_or_path ${model_name_or_path} \
            --task_name ${TASK_NAME} \
            --output_dir ${tuned_checkpoint} \
            --quantization_approach ${approach} \
            --do_train \
            --tune \
            --overwrite_output_dir \
            --cache_dir ${cache_dir} \
            --worker "${worker}" \
            --task_index ${task_index} \
            ${extra_cmd}
    fi
}

main "$@"
