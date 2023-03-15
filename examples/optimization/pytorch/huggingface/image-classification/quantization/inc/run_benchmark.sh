#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  topology="vit-base-patch16-224-static"
  batch_size=8
  tuned_checkpoint="saved_results"
  script="run_image_classification.py"
  mode="benchmark"
  DATASET_NAME="imagenet-1k"
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
    extra_cmd="--do_eval --no_cuda --overwrite_output_dir"

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

    if [ "${topology}" = "vit-base-patch16-224_static" ]; then
        model_name_or_path="/tf_dataset2/models/nlp_toolkit/vit-base"
        extra_cmd=$extra_cmd" --dataset_name ${DATASET_NAME} --load_dataset_from_file ${dataset_location}"
    fi

    if [[ ${int8} == "true" ]]; then
        extra_cmd=$extra_cmd" --int8"
        model_name_or_path=${tuned_checkpoint}
    fi
    echo $extra_cmd

    python -u ${script} \
        --model_name_or_path ${model_name_or_path} \
        --per_device_eval_batch_size ${batch_size} \
        --remove_unused_columns False \
        --output_dir ./tmp/benchmark_output \
        ${mode_cmd} \
        ${extra_cmd}
}

main "$@"
