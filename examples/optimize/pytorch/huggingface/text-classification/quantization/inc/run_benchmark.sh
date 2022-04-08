#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  iters=100
  batch_size=16
  tuned_checkpoint=saved_results
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
      --mode=*)
          mode=$(echo $var |cut -f2 -d=)
      ;;
      --batch_size=*)
          batch_size=$(echo $var |cut -f2 -d=)
      ;;
      --iters=*)
          iters=$(echo ${var} |cut -f2 -d=)
      ;;
      --int8=*)
          int8=$(echo ${var} |cut -f2 -d=)
      ;;
      --config=*)
          tuned_checkpoint=$(echo $var |cut -f2 -d=)
      ;;
      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;
    esac
  done

}


# run_benchmark
function run_benchmark {
    extra_cmd=''
    MAX_SEQ_LENGTH=128

    if [[ ${mode} == "accuracy" ]]; then
        mode_cmd=" --accuracy_only"
    elif [[ ${mode} == "benchmark" ]]; then
        mode_cmd=" --benchmark "
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi

    if [ "${topology}" = "bert_base_mrpc_qat" ]; then
        TASK_NAME="mrpc"
        model_name_or_path="textattack/bert-base-uncased-MRPC"
        model_type="bert"
        approach="QuantizationAwareTraining"
        extra_cmd=$extra_cmd" --learning_rate 2e-5 \
                   --num_train_epochs 3 \
                   --eval_steps 100 \
                   --save_steps 100 \
                   --greater_is_better True \
                   --load_best_model_at_end True \
                   --evaluation_strategy steps \
                   --save_strategy steps \
                   --metric_for_best_model accuracy \
                   --save_total_limit 1"
    elif [ "${topology}" = "bert_base_mrpc_static" ]; then
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

    if [[ ${int8} == "true" ]]; then
        extra_cmd=$extra_cmd" --int8"
    fi
    echo $extra_cmd

    python -u run_glue.py \
        --model_name_or_path ${tuned_checkpoint} \
        --task_name ${TASK_NAME} \
        --do_eval \
        --max_seq_length ${MAX_SEQ_LENGTH} \
        --per_device_eval_batch_size ${batch_size} \
        --output_dir ./tmp/benchmark_output \
        --overwrite_output_dir \
        --no_cuda \
        ${mode_cmd} \
        ${extra_cmd}
}

main "$@"
