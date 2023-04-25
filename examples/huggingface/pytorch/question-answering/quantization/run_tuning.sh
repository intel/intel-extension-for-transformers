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
    extra_cmd = ''
    framework=$(echo $topology | grep "ipex")
    if [[ "$framework" != "" ]];then
        extra_cmd=$extra_cmd" --framework ipex"
    fi
    if [ "${topology}" = "distilbert_base_squad_static" ]; then
        DATASET_NAME="squad"
        model_name_or_path="distilbert-base-uncased-distilled-squad"
        approach="PostTrainingStatic"
    elif [ "${topology}" = "distilbert_base_squad_dynamic" ]; then
        DATASET_NAME="squad"
        model_name_or_path="distilbert-base-uncased-distilled-squad"
        approach="PostTrainingDynamic"
    elif [ "${topology}" = "distilbert_base_squad_qat" ]; then
        DATASET_NAME="squad"
        model_name_or_path="distilbert-base-uncased-distilled-squad"
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
    elif [ "${topology}" = "bert_large_SQuAD_static" ]; then
        DATASET_NAME="squad"
        model_name_or_path="bert-large-uncased-whole-word-masking-finetuned-squad"
        approach="PostTrainingStatic"
    elif [ "${topology}" = "roberta_base_SQuAD2_static" ]; then
        DATASET_NAME="squad"
        model_name_or_path="deepset/roberta-base-squad2"
        approach="PostTrainingStatic"
        # extra_cmd=$extra_cmd" --version_2_with_negative"
    elif [ "${topology}" = "longformer_base_squad_static" ]; then
        DATASET_NAME="squad"
        model_name_or_path="valhalla/longformer-base-4096-finetuned-squadv1"
        approach="PostTrainingStatic"
        extra_cmd=$extra_cmd" --strategy mse_v2"
    elif [ "${topology}" = "longformer_base_squad_dynamic" ]; then
        DATASET_NAME="squad"
        model_name_or_path="valhalla/longformer-base-4096-finetuned-squadv1"
        approach="PostTrainingDynamic"
        extra_cmd=$extra_cmd" --strategy mse_v2"
    elif [ "${topology}" = "distilbert_base_squad_ipex" ]; then
        DATASET_NAME="squad"
        model_name_or_path="distilbert-base-uncased-distilled-squad"
        extra_cmd=$extra_cmd" --perf_tol 0.02"
        approach="PostTrainingStatic"
    elif [ "${topology}" = "bert_large_squad_ipex" ]; then
        DATASET_NAME="squad"
        model_name_or_path="bert-large-uncased-whole-word-masking-finetuned-squad"
        approach="PostTrainingStatic"
    fi

    python -u ./run_qa.py \
        --model_name_or_path ${model_name_or_path} \
        --dataset_name ${DATASET_NAME} \
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
