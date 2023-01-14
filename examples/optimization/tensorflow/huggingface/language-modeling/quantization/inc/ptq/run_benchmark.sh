#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  topology="distilgpt2_clm"
  iters=100
  batch_size=16
  tuned_checkpoint=saved_results
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

    if [ "${topology}" = "distilgpt2_clm" ]; then
        script="run_clm.py"
        dataset_name="wikitext"
        model_name_or_path="distilgpt2"
        dataset_config_name="wikitext-2-raw-v1"
        # remove following two parameters if you have enough memory
        extra_cmd=$extra_cmd" --max_eval_samples 196 --block_size 128"
    elif [ "${topology}" = "distilbert_mlm" ]; then
        script="run_mlm.py"
        dataset_name="wikitext"
        model_name_or_path="distilbert-base-cased"
        dataset_config_name="wikitext-2-raw-v1"
        # remove following two parameters if you have enough memory
        extra_cmd=$extra_cmd" --max_eval_samples 196 --max_seq_length 128"
    elif [ "${topology}" = "distilroberta_mlm" ]; then
        script="run_mlm.py"
        dataset_name="wikitext"
        model_name_or_path="Rocketknight1/distilroberta-base-finetuned-wikitext2"
        dataset_config_name="wikitext-2-raw-v1"
        # remove following two parameters if you have enough memory
        extra_cmd=$extra_cmd" --max_eval_samples 196 --max_seq_length 128"
    fi

    if [[ ${int8} == "true" ]]; then
        extra_cmd=$extra_cmd" --int8"
    fi
    echo $extra_cmd

    python -u ../${script} \
        --model_name_or_path ${model_name_or_path} \
        --dataset_name ${dataset_name} \
        --dataset_config_name ${dataset_config_name} \
        --do_eval \
        --per_device_eval_batch_size ${batch_size} \
        --output_dir ${tuned_checkpoint} \
        --overwrite_output_dir \
        --cache_dir ${cache_dir} \
        ${mode_cmd} \
        ${extra_cmd}
}

main "$@"
