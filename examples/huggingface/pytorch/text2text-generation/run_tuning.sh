#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {
  topology="flan-t5"
  tuned_checkpoint="saved_results"
  DATASET_NAME="NeelNanda/pile-10k"
  model_name_or_path="google/flan-t5-large"
  extra_cmd=""
  batch_size=8
  approach="PostTrainingStatic"
  alpha=0.6
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
       --task=*)
           task=$(echo $var |cut -f2 -d=)
       ;;
       --approach=*)
           approach=$(echo $var |cut -f2 -d=)
       ;;
       --backend=*)
           backend=$(echo $var |cut -f2 -d=)
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

    script="run_seq2seq_generation.py"
    if [ "${topology}" = "flan-t5" ]; then
        model_type="t5"
        model_name_or_path="google/flan-t5-large"
        if [ "${backend}" = "ipex" ]; then
            extra_cmd=$extra_cmd" --ipex"
            alpha=0.7
        fi

    fi
    if [ "${topology}" = "t5-base-tag" ]; then
        model_type="t5"
        model_name_or_path="fabiochiu/t5-base-tag-generation"
        if [ "${backend}" = "ipex" ]; then
            extra_cmd=$extra_cmd" --ipex"
            alpha=0.7
        fi

    fi

    if [ ${script} = "run_seq2seq_generation.py" ];then
        python -u ./${script} \
            --model_type ${model_type} \
            --model_name_or_path ${model_name_or_path} \
            --output_dir ${tuned_checkpoint} \
            --dataset ${DATASET_NAME} \
            --quantize \
            --sq \
            --alpha ${alpha} \
            ${extra_cmd}
    else
        echo "Error: Please provide the correct script."
        exit 1
    fi
}

main "$@"
