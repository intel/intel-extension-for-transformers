#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {
  topology="bloom"
  tuned_checkpoint="saved_results"
  tasks="lambada"
  model_name_or_path="bigscience/bloom-560m"
  extra_cmd=""
  batch_size=8
  max_cali_sample=100
  max_train_sample=100


  model_type="bloom"
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
    if [ "${topology}" = "bloom_text_static" ]; then
        script="run_text.py"
        DATASET_NAME="lambada"
        model_name_or_path="bigscience/bloom-560m"
        approach="PostTrainingStatic"
        model_type="bloom"
    elif [ "${topology}" = "bloom_text_dynamic" ]; then
        script="run_text.py"
        DATASET_NAME="lambada"
        model_name_or_path="bigscience/bloom-560m"
        approach="PostTrainingDynamic"
        model_type="bloom"
    fi
    python -u ./${script} \
        --model_name_or_path ${model_name_or_path} \
        --tasks ${DATASET_NAME} \
        --do_eval \
        --do_train \
        --model_type ${model_type} \
        --max_cali_sample ${max_cali_sample} \
        --max_train_sample ${max_train_sample} \
        --per_device_eval_batch_size ${batch_size} \
        --output_dir ${tuned_checkpoint} \
        --tune \
        --overwrite_output_dir \
        --overwrite_cache \
        --quantization_approach ${approach} \
        ${extra_cmd}
}

main "$@"
