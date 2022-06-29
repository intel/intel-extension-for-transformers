#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {
  topology="gpt"
  tuned_checkpoint="saved_results"
  DATASET_NAME="wikitext"
  DATASET_CONFIG_NAME="wikitext-2-raw-v1"
  model_name_or_path="EleutherAI/gpt-neo-125M"
  extra_cmd=""
  batch_size=8
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
    if [ "${topology}" = "gpt_neo_clm_static" ]; then
        script="run_clm.py"
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="EleutherAI/gpt-neo-125M"
        approach="PostTrainingStatic"
    elif [ "${topology}" = "gpt_neo_clm_dynamic" ]; then
        script="run_clm.py"
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="EleutherAI/gpt-neo-125M"
        approach="PostTrainingDynamic"
    elif [ "${topology}" = "bert_mlm_static" ]; then
        script="run_mlm.py"
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="bert-base-uncased"
        approach="PostTrainingStatic"
    elif [ "${topology}" = "bert_mlm_dynamic" ]; then
        script="run_mlm.py"
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="bert-base-uncased"
        approach="PostTrainingDynamic"
    elif [ "${topology}" = "xlnet_plm_static" ]; then
        script="run_plm.py"
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="xlnet-base-cased"
        approach="PostTrainingStatic"
    elif [ "${topology}" = "xlnet_plm_dynamic" ]; then
        script="run_plm.py"
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="xlnet-base-cased"
        approach="PostTrainingDynamic"
    elif [ "${topology}" = "reformer_crime_and_punishment_static" ]; then
        script="run_clm.py"
        DATASET_NAME="crime_and_punish"
        model_name_or_path="google/reformer-crime-and-punishment"
        approach="PostTrainingStatic"
    elif [ "${topology}" = "ctrl_wikitext_static" ]; then
        script="run_clm.py"
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="sshleifer/tiny-ctrl"
        approach="PostTrainingStatic"
    fi
    python -u ./${script} \
        --model_name_or_path ${model_name_or_path} \
        --dataset_name ${DATASET_NAME} \
        --dataset_config_name ${DATASET_CONFIG_NAME} \
        --do_eval \
        --do_train \
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
