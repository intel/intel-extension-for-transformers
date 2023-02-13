#!/bin/bash
# set -x
function main {
  init_params "$@"
  prepare_model
}

# init params
function init_params {
  for var in "$@"
  do
    case $var in
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --dataset_name=*)
          dataset_name=$(echo $var |cut -f2 -d=)
      ;;
      --cache_dir=*)
          cache_dir=$(echo $var |cut -f2 -d=)
      ;;
      --output_dir=*)
          output_dir=$(echo $var |cut -f2 -d=)
      ;;
      --precision=*)
          precision=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done
}

function prepare_model {

    mode_cmd=""
    if [[ ${precision} = 'int8' ]]; then
        mode_cmd=$mode_cmd"--do_train --tune --quantization_approach PostTrainingStatic"
    fi
    if [[ ${precision} = 'bf16' ]]; then
        mode_cmd=$mode_cmd" --enable_bf16"
    fi
    echo ${mode_cmd}
   
    cache="./tmp"
    if [[ ${cache_dir} ]]; then
        cache="$cache_dir"
    fi
    echo ${cache}
 
    python run_qa.py \
        --model_name_or_path ${input_model} \
        --dataset_name ${dataset_name} \
        --do_eval \
        --cache_dir ${cache} \
        --output_dir ${output_dir} \
        --overwrite_output_dir \
        --to_onnx ./model_and_tokenizer/fp32-model.onnx \
        ${mode_cmd}

}

main "$@"

