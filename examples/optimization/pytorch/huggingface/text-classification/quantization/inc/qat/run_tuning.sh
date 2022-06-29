#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {
  topology="bert_base_SST-2"
  tuned_checkpoint="saved_results"
  TASK_NAME="mrpc"
  model_name_or_path="bert-base-cased"
  extra_cmd=""
  batch_size=8
  MAX_SEQ_LENGTH=128
  model_type="bert"
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
    if [ "${topology}" = "bert_base_mrpc" ]; then
        TASK_NAME="mrpc"
        model_name_or_path="bert-base-uncased"
        model_type="bert"
        approach="QuantizationAwareTraining"
        extra_cmd=$extra_cmd" --learning_rate 1e-5 \
                   --num_train_epochs 6 \
                   --eval_steps 100 \
                   --save_steps 100 \
                   --greater_is_better True \
                   --load_best_model_at_end True \
                   --evaluation_strategy steps \
                   --save_strategy steps \
                   --metric_for_best_model accuracy \
                   --save_total_limit 1"
    fi


    python -u ../run_glue.py \
        --model_name_or_path ${model_name_or_path} \
        --task_name ${TASK_NAME} \
        --do_eval \
        --do_train \
        --max_seq_length ${MAX_SEQ_LENGTH} \
        --per_device_eval_batch_size ${batch_size} \
        --output_dir ${tuned_checkpoint} \
        --no_cuda \
        --tune \
        --overwrite_output_dir \
        --overwrite_cache \
        --quantization_approach ${approach} \
        ${extra_cmd}
}

main "$@"
