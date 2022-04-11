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
    if [ "${topology}" = "bert_base_mrpc_static" ]; then
        TASK_NAME="mrpc"
        model_name_or_path="textattack/bert-base-uncased-MRPC"
        approach="PostTrainingStatic"
    elif [ "${topology}" = "bert_base_mrpc_dynamic" ]; then
        TASK_NAME="mrpc"
        model_name_or_path="textattack/bert-base-uncased-MRPC"
        approach="PostTrainingDynamic"
    elif [ "${topology}" = "bert_base_SST-2_static" ]; then
        TASK_NAME='sst2'
        model_name_or_path="echarlaix/bert-base-uncased-sst2-acc91.1-d37-hybrid"
        approach="PostTrainingStatic"
    elif [ "${topology}" = "bert_base_SST-2_dynamic" ]; then
        TASK_NAME='sst2'
        model_name_or_path="echarlaix/bert-base-uncased-sst2-acc91.1-d37-hybrid"
        approach="PostTrainingDynamic"
    elif [ "${topology}" = "distillbert_base_SST-2_static" ]; then
        TASK_NAME='sst2'
        model_name_or_path="distilbert-base-uncased-finetuned-sst-2-english"
        approach="PostTrainingStatic"
    elif [ "${topology}" = "distillbert_base_SST-2_dynamic" ]; then
        TASK_NAME='sst2'
        model_name_or_path="distilbert-base-uncased-finetuned-sst-2-english"
        approach="PostTrainingDynamic"
    elif [ "${topology}" = "albert_base_MRPC_static" ]; then
        TASK_NAME='MRPC'
        model_name_or_path="textattack/albert-base-v2-MRPC" 
        model_type='albert'
        approach="PostTrainingStatic"
    elif [ "${topology}" = "albert_base_MRPC_dynamic" ]; then
        TASK_NAME='MRPC'
        model_name_or_path="textattack/albert-base-v2-MRPC" 
        model_type='albert'
        approach="PostTrainingDynamic"
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
        --quantization_approach ${approach} \
        ${extra_cmd}
}

main "$@"
