#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {
  topology="pegasus_samsum_dynamic"
  tuned_checkpoint="saved_results"
  DATASET_NAME="xsum"
  extra_cmd=""
  batch_size=8
  approach="PostTrainingStatic"
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
      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;
    esac
  done

}

# run_tuning
function run_tuning {
    if [ "${topology}" = "pegasus_samsum_dynamic" ]; then
        DATASET_NAME="samsum"
        model_name_or_path="lvwerra/pegasus-samsum"
        approach="PostTrainingDynamic"
    else
        echo "unsupport topology: ${topology}"
        exit 1
    fi
    python -u ./run_summarization.py \
        --model_name_or_path ${model_name_or_path} \
        --dataset_name ${DATASET_NAME} \
        --do_eval \
        --do_train \
        --per_device_eval_batch_size ${batch_size} \
        --output_dir ${tuned_checkpoint} \
        --no_cuda \
        --tune \
        --overwrite_output_dir \
        --overwrite_cache \
        --predict_with_generate \
        --quantization_approach ${approach} \
        ${extra_cmd}
}

main "$@"
