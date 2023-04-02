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
  script="../run_glue.py"
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
    extra_cmd="--do_eval --max_seq_length 128 --no_cuda --overwrite_output_dir --overwrite_cache"

    if [[ ${mode} == "accuracy" ]]; then
        mode_cmd=" --accuracy_only"
    elif [[ ${mode} == "benchmark" ]]; then
        mode_cmd=" --benchmark "
    elif [[ ${mode} == "benchmark_only" ]]; then
        mode_cmd=" --benchmark_only "
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi

    if [ "${topology}" = "bert_base_mrpc_static" ]; then
        model_name_or_path="textattack/bert-base-uncased-MRPC"
        extra_cmd=$extra_cmd" --task_name mrpc"
    elif [ "${topology}" = "bert_base_mrpc_dynamic" ]; then
        extra_cmd=$extra_cmd" --task_name mrpc"
        model_name_or_path="textattack/bert-base-uncased-MRPC"
    elif [ "${topology}" = "bert_base_SST-2_static" ]; then
        extra_cmd=$extra_cmd" --task_name sst2"
        model_name_or_path="echarlaix/bert-base-uncased-sst2-acc91.1-d37-hybrid"
    elif [ "${topology}" = "bert_base_SST-2_dynamic" ]; then
        extra_cmd=$extra_cmd" --task_name sst2"
        model_name_or_path="echarlaix/bert-base-uncased-sst2-acc91.1-d37-hybrid"
    elif [ "${topology}" = "bert_base_CoLA_static" ]; then
        extra_cmd=$extra_cmd" --task_name cola"
        model_name_or_path="textattack/bert-base-uncased-CoLA"
    elif [ "${topology}" = "bert_base_STS-B_static" ]; then
        extra_cmd=$extra_cmd" --task_name stsb"
        model_name_or_path="Contrastive-Tension/BERT-Base-CT-STSb"
    elif [ "${topology}" = "bert_base_RTE_static" ]; then
        extra_cmd=$extra_cmd" --task_name rte"
        model_name_or_path="textattack/bert-base-uncased-RTE"
    elif [ "${topology}" = "bert_large_RTE_static" ]; then
        extra_cmd=$extra_cmd" --task_name rte"
        model_name_or_path="yoshitomo-matsubara/bert-large-uncased-rte"
    elif [ "${topology}" = "bert_large_CoLA_static" ]; then
        extra_cmd=$extra_cmd" --task_name cola"
        model_name_or_path="yoshitomo-matsubara/bert-large-uncased-cola"
    elif [ "${topology}" = "bert_large_MRPC_static" ]; then
        extra_cmd=$extra_cmd" --task_name mrpc"
        model_name_or_path="yoshitomo-matsubara/bert-large-uncased-mrpc"
    elif [ "${topology}" = "bert_large_QNLI_static" ]; then
        extra_cmd=$extra_cmd" --task_name qnli"
        model_name_or_path="yoshitomo-matsubara/bert-large-uncased-qnli"
    elif [ "${topology}" = "camembert_base_XNLI_dynamic" ]; then
        model_name_or_path="BaptisteDoyen/camembert-base-xnli"
        extra_cmd=$extra_cmd" --dataset_name xnli --dataset_config_name fr"
    elif [ "${topology}" = "xlnet_base_SST-2_static" ]; then
        extra_cmd=$extra_cmd" --task_name sst2"
        model_name_or_path="textattack/xlnet-base-cased-SST-2"
    elif [ "${topology}" = "funnel_small_MRPC_static" ]; then
        extra_cmd=$extra_cmd" --task_name mrpc"
        model_name_or_path="funnel-transformer/small-base"
    elif [ "${topology}" = "roberta_base_SST-2_dynamic" ]; then
        extra_cmd=$extra_cmd" --task_name sst2"
        model_name_or_path="textattack/roberta-base-SST-2"
    elif [ "${topology}" = "distillbert_base_SST-2_static" ]; then
        extra_cmd=$extra_cmd" --task_name sst2"
        model_name_or_path="distilbert-base-uncased-finetuned-sst-2-english"
    elif [ "${topology}" = "distillbert_base_SST-2_dynamic" ]; then
        extra_cmd=$extra_cmd" --task_name sst2"
        model_name_or_path="distilbert-base-uncased-finetuned-sst-2-english"
    elif [ "${topology}" = "albert_base_MRPC_static" ]; then
        extra_cmd=$extra_cmd" --task_name mrpc"
        model_name_or_path="textattack/albert-base-v2-MRPC" 
    elif [ "${topology}" = "albert_base_MRPC_dynamic" ]; then
        extra_cmd=$extra_cmd" --task_name mrpc"
        model_name_or_path="textattack/albert-base-v2-MRPC" 
    elif [ "${topology}" = "xlm_roberta_large_XNLI_dynamic" ]; then
        extra_cmd=$extra_cmd" --dataset_name xnli --dataset_config_name en"
        model_name_or_path="joeddav/xlm-roberta-large-xnli"
    elif [ "${topology}" = "bert_base_SST-2_static_no_trainer" ]; then
        extra_cmd=" --task_name sst2"
        model_name_or_path="echarlaix/bert-base-uncased-sst2-acc91.1-d37-hybrid"
        script="../run_glue_no_trainer.py" 
    fi

    if [[ ${int8} == "true" ]]; then
        extra_cmd=$extra_cmd" --int8"
        model_name_or_path=${tuned_checkpoint}
    fi
    echo $extra_cmd

    python -u ${script} \
        --model_name_or_path ${model_name_or_path} \
        --per_device_eval_batch_size ${batch_size} \
        --output_dir ./tmp/benchmark_output \
        ${mode_cmd} \
        ${extra_cmd}
}

main "$@"
