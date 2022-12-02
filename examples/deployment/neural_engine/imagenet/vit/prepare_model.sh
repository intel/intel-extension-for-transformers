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
      --task_name=*)
          task_name=$(echo $var |cut -f2 -d=)
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
        mode_cmd=$mode_cmd" --tune  "
    fi
    if [[ ${precision} = 'bf16' ]]; then
        mode_cmd=$mode_cmd" --enable_bf16"
    fi
    echo ${mode_cmd}
    mode_cmd_cache=""
    if [[ ${cache_dir} ]]; then
        mode_cmd_cache=$mode_cmd_cache" --load_dataset_from_file ${cache_dir}"
    fi
 
    python -u model_quant_convert.py \
        --model_name_or_path ${input_model} \
        --dataset_name ${task_name} \
        --remove_unused_columns False \
        --do_eval \
        --no_cuda \
        --output_dir ${output_dir} \
        --overwrite_output_dir \
        --accuracy_only \
        --do_train \
        --per_device_eval_batch_size 8 \
        ${mode_cmd} \
        ${mode_cmd_cache}
}

main "$@"

