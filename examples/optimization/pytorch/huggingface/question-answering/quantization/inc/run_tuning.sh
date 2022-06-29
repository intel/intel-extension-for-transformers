#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {
  topology="distilbert_base_squad_static"
  tuned_checkpoint="saved_results"
  DATASET_NAME="squad"
  model_name_or_path="distilbert-base-uncased-distilled-squad"
  extra_cmd=""
  batch_size=8
  MAX_SEQ_LENGTH=384
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
    if [ "${topology}" = "distilbert_base_squad_static" ]; then
        DATASET_NAME="squad"
        model_name_or_path="distilbert-base-uncased-distilled-squad"
        approach="PostTrainingStatic"
    elif [ "${topology}" = "distilbert_base_squad_dynamic" ]; then
        DATASET_NAME="squad"
        model_name_or_path="distilbert-base-uncased-distilled-squad"
        approach="PostTrainingDynamic"
    elif [ "${topology}" = "bert_large_SQuAD_static" ]; then
        DATASET_NAME="squad"
        model_name_or_path="bert-large-uncased-whole-word-masking-finetuned-squad"
        approach="PostTrainingStatic"
    elif [ "${topology}" = "roberta_base_SQuAD2_static" ]; then
        DATASET_NAME="squad"
        model_name_or_path="deepset/roberta-base-squad2"
        approach="PostTrainingStatic"
        # extra_cmd=$extra_cmd" --version_2_with_negative"
    fi

    python -u ./run_qa.py \
        --model_name_or_path ${model_name_or_path} \
        --dataset_name ${DATASET_NAME} \
        --do_eval \
        --do_train \
        --max_seq_length ${MAX_SEQ_LENGTH} \
        --per_device_eval_batch_size ${batch_size} \
        --max_eval_samples 5000 \
        --output_dir ${tuned_checkpoint} \
        --no_cuda \
        --tune \
        --overwrite_output_dir \
        --overwrite_cache \
        --quantization_approach ${approach} \
        ${extra_cmd}
}

main "$@"
