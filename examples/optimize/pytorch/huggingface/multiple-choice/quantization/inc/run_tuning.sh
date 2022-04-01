#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {
  tuned_checkpoint="saved_results"
  approach="PostTrainingStatic"
  batch_size=8
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
    if [ "${topology}" = "bert_base_swag_static" ]; then
        model_name_or_path="ehdwns1516/bert-base-uncased_SWAG"
        model_type="bert"
        approach="PostTrainingStatic"
    elif [ "${topology}" = "bert_base_swag_dynamic" ]; then
        model_name_or_path="ehdwns1516/bert-base-uncased_SWAG"
        model_type="bert"
        approach="PostTrainingDynamic"
    fi

    python -u ./run_swag.py \
        --model_name_or_path ${model_name_or_path} \
        --do_eval \
        --do_train \
        --per_device_eval_batch_size ${batch_size} \
        --per_device_train_batch_size ${batch_size} \
        --output_dir ${tuned_checkpoint} \
        --no_cuda \
        --quantization_approach ${approach} \
        --tune \
        --pad_to_max_length \
        --overwrite_output_dir 
}

main "$@"