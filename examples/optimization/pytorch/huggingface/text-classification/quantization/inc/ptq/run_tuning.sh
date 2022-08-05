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
  extra_cmd="--do_eval --do_train --max_seq_length 128 --no_cuda --overwrite_output_dir --overwrite_cache"
  batch_size=8
  MAX_SEQ_LENGTH=128
  model_type="bert"
  approach="PostTrainingStatic"
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
        model_name_or_path="textattack/bert-base-uncased-MRPC"
        approach="PostTrainingStatic"
        extra_cmd=$extra_cmd" --task_name mrpc"
    elif [ "${topology}" = "bert_base_mrpc_dynamic" ]; then
        extra_cmd=$extra_cmd" --task_name mrpc"
        model_name_or_path="textattack/bert-base-uncased-MRPC"
        approach="PostTrainingDynamic"
    elif [ "${topology}" = "bert_base_SST-2_static" ]; then
        extra_cmd=$extra_cmd" --task_name sst2"
        model_name_or_path="echarlaix/bert-base-uncased-sst2-acc91.1-d37-hybrid"
        approach="PostTrainingStatic"
    elif [ "${topology}" = "bert_base_SST-2_dynamic" ]; then
        extra_cmd=$extra_cmd" --task_name sst2"
        model_name_or_path="echarlaix/bert-base-uncased-sst2-acc91.1-d37-hybrid"
        approach="PostTrainingDynamic"
    elif [ "${topology}" = "bert_base_CoLA_static" ]; then
        extra_cmd=$extra_cmd" --task_name cola"
        model_name_or_path="textattack/bert-base-uncased-CoLA"
        approach="PostTrainingStatic"
    elif [ "${topology}" = "bert_base_STS-B_static" ]; then
        extra_cmd=$extra_cmd" --task_name stsb"
        model_name_or_path="Contrastive-Tension/BERT-Base-CT-STSb"
        approach="PostTrainingStatic"
    elif [ "${topology}" = "bert_base_RTE_static" ]; then
        extra_cmd=$extra_cmd" --task_name rte"
        model_name_or_path="textattack/bert-base-uncased-RTE"
        approach="PostTrainingStatic"
    elif [ "${topology}" = "bert_large_RTE_static" ]; then
        extra_cmd=$extra_cmd" --task_name rte"
        model_name_or_path="yoshitomo-matsubara/bert-large-uncased-rte"
        approach="PostTrainingStatic"
    elif [ "${topology}" = "bert_large_CoLA_static" ]; then
        extra_cmd=$extra_cmd" --task_name cola"
        model_name_or_path="yoshitomo-matsubara/bert-large-uncased-cola"
        approach="PostTrainingStatic"
    elif [ "${topology}" = "bert_large_MRPC_static" ]; then
        extra_cmd=$extra_cmd" --task_name mrpc"
        model_name_or_path="yoshitomo-matsubara/bert-large-uncased-mrpc"
        approach="PostTrainingStatic"
    elif [ "${topology}" = "bert_large_QNLI_static" ]; then
        extra_cmd=$extra_cmd" --task_name qnli"
        model_name_or_path="yoshitomo-matsubara/bert-large-uncased-qnli"
        approach="PostTrainingStatic"
    elif [ "${topology}" = "camembert_base_XNLI_dynamic" ]; then
        model_name_or_path="BaptisteDoyen/camembert-base-xnli"
        approach="PostTrainingDynamic"
        extra_cmd=$extra_cmd" --dataset_name xnli --dataset_config_name fr"
    elif [ "${topology}" = "xlnet_base_SST-2_static" ]; then
        extra_cmd=$extra_cmd" --task_name sst2"
        model_name_or_path="textattack/xlnet-base-cased-SST-2"
        approach="PostTrainingStatic"
    elif [ "${topology}" = "funnel_small_MRPC_static" ]; then
        extra_cmd=$extra_cmd" --task_name mrpc"
        model_name_or_path="funnel-transformer/small-base"
        approach="PostTrainingStatic"
    elif [ "${topology}" = "roberta_base_SST-2_dynamic" ]; then
        extra_cmd=$extra_cmd" --task_name sst2"
        model_name_or_path="textattack/roberta-base-SST-2"
        approach="PostTrainingDynamic"
    elif [ "${topology}" = "distillbert_base_SST-2_static" ]; then
        extra_cmd=$extra_cmd" --task_name sst2"
        model_name_or_path="distilbert-base-uncased-finetuned-sst-2-english"
        approach="PostTrainingStatic"
    elif [ "${topology}" = "distillbert_base_SST-2_dynamic" ]; then
        extra_cmd=$extra_cmd" --task_name sst2"
        model_name_or_path="distilbert-base-uncased-finetuned-sst-2-english"
        approach="PostTrainingDynamic"
    elif [ "${topology}" = "albert_base_MRPC_static" ]; then
        extra_cmd=$extra_cmd" --task_name mrpc"
        model_name_or_path="textattack/albert-base-v2-MRPC" 
        approach="PostTrainingStatic"
    elif [ "${topology}" = "albert_base_MRPC_dynamic" ]; then
        extra_cmd=$extra_cmd" --task_name mrpc"
        model_name_or_path="textattack/albert-base-v2-MRPC" 
        approach="PostTrainingDynamic"
    elif [ "${topology}" = "xlm_roberta_large_XNLI_dynamic" ]; then
        extra_cmd=$extra_cmd" --dataset_name xnli --dataset_config_name en"
        model_name_or_path="joeddav/xlm-roberta-large-xnli" 
        approach="PostTrainingDynamic"
    elif [ "${topology}" = "bert_base_SST-2_static_no_trainer" ]; then
        extra_cmd=" --task_name sst2"
        model_name_or_path="echarlaix/bert-base-uncased-sst2-acc91.1-d37-hybrid"
        approach="PostTrainingStatic"
        script="../run_glue_no_trainer.py"
    fi


    python -u ${script} \
        --model_name_or_path ${model_name_or_path} \
        --per_device_eval_batch_size ${batch_size} \
        --output_dir ${tuned_checkpoint} \
        --tune \
        --quantization_approach ${approach} \
        ${extra_cmd}
}

main "$@"
