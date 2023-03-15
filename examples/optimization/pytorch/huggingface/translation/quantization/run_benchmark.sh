#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  topology="pegasus_xsum_static"
  iters=100
  batch_size=16
  tuned_checkpoint=saved_results
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
    elif [[ ${mode} == "benchmark_only" ]]; then
        mode_cmd=" --benchmark_only "
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi

    if [ "${topology}" = "t5-small_dynamic" ]; then
        model_name_or_path="t5-small"
        extra_cmd=$extra_cmd" --source_lang en --target_lang ro --dataset_name wmt16 --dataset_config_name ro-en"
    elif [ "${topology}" = "marianmt_WMT_en_ro_dynamic" ]; then
        model_name_or_path='Helsinki-NLP/opus-mt-en-ro'
        extra_cmd=$extra_cmd" --source_lang en --target_lang ro --dataset_name wmt16 --dataset_config_name ro-en"
    else
        echo "unsupport topology: ${topology}"
        exit 1
    fi

    if [[ ${int8} == "true" ]]; then
        extra_cmd=$extra_cmd" --int8"
        model_name_or_path=${tuned_checkpoint}
    fi
    if [ "${topology}" = "t5-small_dynamic" ]; then
        python -u ./run_translation.py \
            --model_name_or_path ${model_name_or_path} \
            --do_eval \
            --per_device_eval_batch_size ${batch_size} \
            --output_dir ${tuned_checkpoint} \
            --no_cuda \
            --overwrite_output_dir \
            --overwrite_cache \
            --predict_with_generate \
            --source_prefix "translate English to Romanian: "\
            ${extra_cmd} \
            ${mode_cmd}
    else
        python -u ./run_translation.py \
            --model_name_or_path ${model_name_or_path} \
            --do_eval \
            --per_device_eval_batch_size ${batch_size} \
            --output_dir ${tuned_checkpoint} \
            --no_cuda \
            --overwrite_output_dir \
            --overwrite_cache \
            --predict_with_generate \
            ${extra_cmd} \
            ${mode_cmd}
    fi
}

main "$@"
