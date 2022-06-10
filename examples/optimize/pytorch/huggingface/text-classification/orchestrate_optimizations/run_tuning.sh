#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {
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
    extra_cmd=''
    batch_size=16

    python -u ./run_glue.py \
        --model_name_or_path Intel/distilbert-base-uncased-sparse-90-unstructured-pruneofa \
        --teacher_model_name_or_path distilbert-base-uncased-finetuned-sst-2-english \
        --task_name sst2 \
        --orchestrate_optimizations \
        --do_eval \
        --do_train \
        --per_device_eval_batch_size ${batch_size} \
        --output_dir ${tuned_checkpoint} \
        --overwrite_output_dir \
        ${extra_cmd}
}

main "$@"
