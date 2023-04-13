#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {
  topology="bert_base_ner"
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
    if [ "${topology}" = "bert_base_ner" ]; then
        TASK_NAME="ner"
        model_name_or_path="dslim/bert-base-NER"
        approach="PostTrainingStatic"
        dataset_name=conll2003
    fi
    
    if [ "${worker}" = "" ]
    then
        python -u run_ner.py \
            --model_name_or_path ${model_name_or_path} \
            --dataset_name ${dataset_name} \
            --task_name ${TASK_NAME} \
            --pad_to_max_length \
            --do_eval \
            --max_length ${MAX_SEQ_LENGTH} \
            --per_device_train_batch_size ${batch_size} \
            --per_device_eval_batch_size ${batch_size} \
            --output_dir ${tuned_checkpoint} \
            --no_cuda \
            --overwrite_output_dir \
            --cache_dir ${cache_dir} \
            --quantization_approach ${approach} \
            --tune \
            ${extra_cmd}
    else
        python -u run_ner.py \
            --model_name_or_path ${model_name_or_path} \
            --dataset_name ${dataset_name} \
            --task_name ${TASK_NAME} \
            --pad_to_max_length \
            --do_eval \
            --max_length ${MAX_SEQ_LENGTH} \
            --per_device_train_batch_size ${batch_size} \
            --per_device_eval_batch_size ${batch_size} \
            --output_dir ${tuned_checkpoint} \
            --no_cuda \
            --overwrite_output_dir \
            --cache_dir ${cache_dir} \
            --quantization_approach ${approach} \
            --tune \
            --worker "${worker}" \
            --task_index ${task_index} \
            ${extra_cmd}
    fi
}

main "$@"
