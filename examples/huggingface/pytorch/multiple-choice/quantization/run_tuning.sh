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
        approach="PostTrainingStatic"
    elif [ "${topology}" = "bert_base_swag_dynamic" ]; then
        model_name_or_path="ehdwns1516/bert-base-uncased_SWAG"
        approach="PostTrainingDynamic"
    elif [ "${topology}" = "bert_base_swag_qat" ]; then
        model_name_or_path="ehdwns1516/bert-base-uncased_SWAG"
        approach="QuantizationAwareTraining"
        extra_cmd=$extra_cmd" --learning_rate 1e-5 \
                   --num_train_epochs 6 \
                   --eval_steps 100 \
                   --save_steps 100 \
                   --greater_is_better True \
                   --load_best_model_at_end True \
                   --evaluation_strategy steps \
                   --save_strategy steps \
                   --save_total_limit 1"
    fi

    python -u ./run_swag.py \
        --model_name_or_path ${model_name_or_path} \
        --do_eval \
        --do_train \
        --per_device_eval_batch_size ${batch_size} \
        --per_device_train_batch_size ${batch_size} \
        --max_eval_samples 1000 \
        --output_dir ${tuned_checkpoint} \
        --no_cuda \
        --quantization_approach ${approach} \
        --tune \
        --pad_to_max_length \
        --overwrite_cache \
        --overwrite_output_dir 
}

main "$@"
