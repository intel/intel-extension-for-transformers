#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {
  topology="vit-base-patch16-224-static"
  tuned_checkpoint="saved_results"
  DATASET_NAME="imagenet-1k"
  model_name_or_path="google/vit-base-patch16-224"
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
    if [ "${topology}" = "vit-base-patch16-224_static" ]; then
        model_name_or_path="/tf_dataset2/models/nlp_toolkit/vit-base"
        approach="PostTrainingStatic"
        inc_config_file="vit_config.yaml"
    fi

    python -u ./run_image_classification.py \
        --model_name_or_path ${model_name_or_path} \
        --dataset_name ${DATASET_NAME} \
        --load_dataset_from_file ${dataset_location} \
        --remove_unused_columns False \
        --do_eval \
        --do_train \
        --per_device_eval_batch_size ${batch_size} \
        --output_dir ${tuned_checkpoint} \
        --no_cuda \
        --tune \
        --overwrite_output_dir \
        --quantization_approach ${approach} \
        --inc_config_file ${inc_config_file} \
        ${extra_cmd}
}

main "$@"
