#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  topology="pegasus_samsum_dynamic"
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
        mode_cmd=" --benchmark --max_eval_samples 100"
    elif [[ ${mode} == "benchmark_only" ]]; then
        mode_cmd=" --benchmark_only --max_eval_samples 100"
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi

    if [ "${topology}" == "pegasus_samsum_dynamic" ]; then
        DATASET_NAME="samsum"
        model_name_or_path="lvwerra/pegasus-samsum"
    elif [ "${topology}" == "t5_base_cnn_dynamic" ]; then
        DATASET_NAME="cnn_dailymail"
        model_name_or_path="flax-community/t5-base-cnn-dm"
        pip install transformers==4.26.0
    elif [ "${topology}" == "t5_large_cnn_dynamic" ]; then
        DATASET_NAME="cnn_dailymail"
        model_name_or_path="sysresearch101/t5-large-finetuned-xsum-cnn"
        pip install transformers==4.26.0
    elif [ "${topology}" == "flan_t5_large_samsum_dynamic" ]; then
        DATASET_NAME="samsum"
        model_name_or_path="stacked-summaries/flan-t5-large-stacked-samsum-1024"
        approach="PostTrainingDynamic"
    elif [ "${topology}" == "flan_t5_large_samsum_static" ]; then
        DATASET_NAME="samsum"
        model_name_or_path="stacked-summaries/flan-t5-large-stacked-samsum-1024"
        approach="PostTrainingStatic"
    else
        echo "unsupport topology: ${topology}"
        exit 1
    fi

    if [[ ${int8} == "true" ]]; then
        extra_cmd=$extra_cmd" --int8"
        model_name_or_path=${tuned_checkpoint}
    fi
    echo $extra_cmd

    if [ "${DATASET_NAME}" == "cnn_dailymail" ]; then
        python -u ./run_summarization.py \
            --model_name_or_path ${model_name_or_path} \
            --dataset_name ${DATASET_NAME} \
            --dataset_config "3.0.0" \
            --source_prefix "summarize: " \
            --do_eval \
            --per_device_eval_batch_size ${batch_size} \
            --output_dir ${tuned_checkpoint} \
            --no_cuda \
            --overwrite_output_dir \
            --overwrite_cache \
            --predict_with_generate \
            ${mode_cmd} \
            ${extra_cmd}
    else
        python -u ./run_summarization.py \
            --model_name_or_path ${model_name_or_path} \
            --dataset_name ${DATASET_NAME} \
            --do_eval \
            --per_device_eval_batch_size ${batch_size} \
            --output_dir ${tuned_checkpoint} \
            --no_cuda \
            --overwrite_output_dir \
            --overwrite_cache \
            --predict_with_generate \
            ${mode_cmd} \
            ${extra_cmd}
    fi
}

main "$@"
