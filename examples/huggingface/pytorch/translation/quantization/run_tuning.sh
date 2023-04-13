#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {
  topology="pegasus_xsum_static"
  tuned_checkpoint="saved_results"
  DATASET_NAME="xsum"
  extra_cmd=""
  batch_size=8
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
    if [ "${topology}" = "t5-small_dynamic" ]; then
        model_name_or_path="t5-small"
        extra_cmd=$extra_cmd" --source_lang en --target_lang ro --dataset_name wmt16 --dataset_config_name ro-en"
        approach="PostTrainingDynamic"
    elif [ "${topology}" = "marianmt_WMT_en_ro_dynamic" ]; then
        model_name_or_path='Helsinki-NLP/opus-mt-en-ro'
        extra_cmd=$extra_cmd" --source_lang en --target_lang ro --dataset_name wmt16 --dataset_config_name ro-en"
        approach="PostTrainingDynamic"
    else
        echo "unsupport topology: ${topology}"
        exit 1
    fi
    if [ "${topology}" = "t5-small_dynamic" ]; then
        python -u ./run_translation.py \
            --model_name_or_path ${model_name_or_path} \
            --do_eval \
            --do_train \
            --per_device_eval_batch_size ${batch_size} \
            --output_dir ${tuned_checkpoint} \
            --no_cuda \
            --tune \
            --overwrite_output_dir \
            --overwrite_cache \
            --predict_with_generate \
            --quantization_approach ${approach} \
            --source_prefix "translate English to Romanian: "\
            ${extra_cmd}
    else
        python -u ./run_translation.py \
            --model_name_or_path ${model_name_or_path} \
            --do_eval \
            --do_train \
            --per_device_eval_batch_size ${batch_size} \
            --output_dir ${tuned_checkpoint} \
            --no_cuda \
            --tune \
            --overwrite_output_dir \
            --overwrite_cache \
            --predict_with_generate \
            --quantization_approach ${approach} \
            ${extra_cmd}
    fi
    
}

main "$@"
