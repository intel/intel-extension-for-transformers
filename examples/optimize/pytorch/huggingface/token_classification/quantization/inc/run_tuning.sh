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
  DATASET_NAME="conll2003"
  model_name_or_path="elastic/distilbert-base-uncased-finetuned-conll03-english "
  extra_cmd=""
  batch_size=8
  MAX_SEQ_LENGTH=384
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
    if [ "${topology}" = "distilbert_base_ner_static" ]; then
        DATASET_NAME="conll2003"
        model_name_or_path="elastic/distilbert-base-uncased-finetuned-conll03-english "
        model_type="bert"
        approach="PostTrainingStatic"
    elif [ "${topology}" = "distilbert_base_ner_dynamic" ]; then
        DATASET_NAME="conll2003"
        model_name_or_path="elastic/distilbert-base-uncased-finetuned-conll03-english "
        model_type="bert"
        approach="PostTrainingDynamic"
    fi

    python -u ./run_ner.py \
        --model_name_or_path ${model_name_or_path} \
        --dataset_name ${DATASET_NAME} \
        --do_eval \
        --do_train \
        --pad_to_max_length \
        --per_device_eval_batch_size ${batch_size} \
        --output_dir ${tuned_checkpoint} \
        --no_cuda \
        --tune \
        --overwrite_output_dir \
        --quantization_approach ${approach} \
        ${extra_cmd}
}

main "$@"
