#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {
  approach="static"
  script="run_whisper.py"
  dataset_location=$HOME/.cache/huggingface
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
      --output_model=*)
          output_model=$(echo $var |cut -f2 -d=)
      ;;
      --approach=*)
          approach=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}

# run_tuning
function run_tuning {

    python -u ${script} \
        --model_name_or_path ${config} \
        --input_model ${input_model} \
        --output_model ${output_model} \
        --cache_dir ${dataset_location} \
        --tune \
        --approach ${approach}

}

main "$@"
