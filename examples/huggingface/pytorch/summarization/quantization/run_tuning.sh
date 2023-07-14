#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {
  topology="pegasus_samsum_dynamic"
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
    if [ "${topology}" == "pegasus_samsum_dynamic" ]; then
        DATASET_NAME="samsum"
        model_name_or_path="lvwerra/pegasus-samsum"
        approach="PostTrainingDynamic"
    elif [ "${topology}" == "t5_base_cnn_dynamic" ]; then
        DATASET_NAME="cnn_dailymail"
        model_name_or_path="flax-community/t5-base-cnn-dm"
        approach="PostTrainingDynamic"
        pip install transformers==4.26.0
    elif [ "${topology}" == "t5_large_cnn_dynamic" ]; then
        DATASET_NAME="cnn_dailymail"
        model_name_or_path="sysresearch101/t5-large-finetuned-xsum-cnn"
        approach="PostTrainingDynamic"
        pip install transformers==4.26.0
    elif [ "${topology}" == "flan_t5_large_samsum_dynamic" ]; then
        DATASET_NAME="samsum"
        model_name_or_path="stacked-summaries/flan-t5-large-stacked-samsum-1024"
        approach="PostTrainingDynamic"
        extra_cmd=$extra_cmd" --perf_tol 0.03"
    elif [ "${topology}" == "flan_t5_large_samsum_static" ]; then
        DATASET_NAME="samsum"
        model_name_or_path="stacked-summaries/flan-t5-large-stacked-samsum-1024"
        approach="PostTrainingStatic"
        extra_cmd=$extra_cmd" --perf_tol 0.03"
    else
        echo "unsupport topology: ${topology}"
        exit 1
    fi

    if [ "${DATASET_NAME}" == "cnn_dailymail" ]; then
        python -u ./run_summarization.py \
            --model_name_or_path ${model_name_or_path} \
            --dataset_name ${DATASET_NAME} \
            --dataset_config "3.0.0" \
            --source_prefix "summarize: " \
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
    else
        python -u ./run_summarization.py \
            --model_name_or_path ${model_name_or_path} \
            --dataset_name ${DATASET_NAME} \
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
