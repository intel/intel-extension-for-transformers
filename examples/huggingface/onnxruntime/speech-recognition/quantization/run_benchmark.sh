#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  iters=100
  dataset_location=$HOME/.cache/huggingface
  script="run_whisper.py"
  for var in "$@"
  do
    case $var in
      --config=*)
          config=$(echo $var |cut -f2 -d=)
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
      --iters=*)
          iters=$(echo ${var} |cut -f2 -d=)
      ;;
      --cores_per_instance=*)
          cores_per_instance=$(echo $var |cut -f2 -d=)
      ;;
      --max_new_tokens=*)
          max_new_tokens=$(echo $var |cut -f2 -d=)
      ;;
      --int8=*)
          int8=$(echo ${var} |cut -f2 -d=)
      ;;
    esac
  done

}


# run_benchmark
function run_benchmark {

    if [[ ${int8} == "false" ]]; then
        input_model=${config}
    fi

    if [[ ${mode} == "accuracy" ]]; then
        mode_cmd=" --accuracy_only"
    elif [[ ${mode} == "benchmark" ]]; then
        mode_cmd=" --benchmark"
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi


    python -u ${script} \
        --model_name_or_path ${config} \
        --cache_dir ${dataset_location} \
        --cores_per_instance ${cores_per_instance-4} \
        --input_model ${input_model} \
        --max_new_tokens ${max_new_tokens-16} \
        --iters ${iters} \
        ${mode_cmd}
}

main "$@"
