#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {
  topology="distilgpt2_clm"
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
    if [ "${topology}" = "distilgpt2_clm" ]; then
        script="run_clm.py"
        model_name_or_path="distilgpt2"
        dataset_name="wikitext"
        approach="PostTrainingStatic"
        dataset_config_name="wikitext-2-raw-v1"
        # remove or change following two parameters if you have enough memory
        extra_cmd=$extra_cmd" --max_eval_samples 96 --block_size 128 --perf_tol 0.08"
    elif [ "${topology}" = "distilbert_mlm" ]; then
        script="run_mlm.py"
        model_name_or_path="distilbert-base-cased"
        dataset_name="wikitext"
        approach="PostTrainingStatic"
        dataset_config_name="wikitext-2-raw-v1"
        # remove or change following two parameters if you have enough memory
        extra_cmd=$extra_cmd" --max_eval_samples 96 --max_seq_length 128 --perf_tol 0.08"
    elif [ "${topology}" = "distilroberta_mlm" ]; then
        script="run_mlm.py"
        model_name_or_path="Rocketknight1/distilroberta-base-finetuned-wikitext2"
        dataset_name="wikitext"
        approach="PostTrainingStatic"
        dataset_config_name="wikitext-2-raw-v1"
        # remove or change following two parameters if you have enough memory
        extra_cmd=$extra_cmd" --max_eval_samples 96 --max_seq_length 128 --perf_tol 0.08"
    fi

    if [ "${worker}" = "" ]
    then
        python -u ../${script} \
            --model_name_or_path ${model_name_or_path} \
            --dataset_name ${dataset_name} \
            --dataset_config_name ${dataset_config_name} \
            --do_eval \
            --output_dir ${tuned_checkpoint} \
            --quantization_approach ${approach} \
            --do_train \
            --overwrite_output_dir \
            --cache_dir ${cache_dir} \
            --tune \
            ${extra_cmd}
    else
        python -u ../${script} \
            --model_name_or_path ${model_name_or_path} \
            --dataset_name ${dataset_name} \
            --dataset_config_name ${dataset_config_name} \
            --task_name ${TASK_NAME} \
            --do_eval \
            --output_dir ${tuned_checkpoint} \
            --quantization_approach ${approach} \
            --do_train \
            --overwrite_output_dir \
            --cache_dir ${cache_dir} \
            --tune \
            --worker "${worker}" \
            --task_index ${task_index} \
            ${extra_cmd}
    fi
}

main "$@"
