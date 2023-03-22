#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  iters=100
  batch_size=8
  tuned_checkpoint=saved_results
  max_cali_sample=100
  max_train_sample=100
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

    if [[ ${mode} == "accuracy" ]]; then
        mode_cmd=" --accuracy_only"
    elif [[ ${mode} == "benchmark" ]]; then
        mode_cmd=" --benchmark "
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi

    if [ "${topology}" = "bloom_text_static" ]; then
        script="run_text.py"
        DATASET_NAME="lambada"
        model_name_or_path="bigscience/bloom-560m"
        approach="PostTrainingStatic"
        model_type="bloom"
    elif [ "${topology}" = "bloom_text_dynamic" ]; then
        script="run_text.py"
        DATASET_NAME="lambada"
        model_name_or_path="bigscience/bloom-560m"
        approach="PostTrainingDynamic"
        model_type="bloom"
    fi
    
    if [[ ${int8} == "true" ]]; then
        extra_cmd=$extra_cmd" --int8"
        model_name_or_path=${tuned_checkpoint}
    fi
    echo $extra_cmd

    python -u ${script} \
        --model_name_or_path ${model_name_or_path} \
        --tasks ${DATASET_NAME} \
        --do_eval \
        --max_cali_sample ${max_cali_sample} \
        --max_train_sample ${max_train_sample} \
        --per_device_eval_batch_size ${batch_size} \
        --output_dir ./tmp/benchmark_output \
        --overwrite_output_dir \
        --overwrite_cache \
        --no_cuda \
        ${mode_cmd} \
        ${extra_cmd}
}

main "$@"