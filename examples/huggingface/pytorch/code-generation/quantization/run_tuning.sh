#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {
  topology="gpt"
  tuned_checkpoint="saved_results"
  DATASET_NAME="mbpp"
  model_name_or_path="bigcode/starcoder"
  extra_cmd=""
  batch_size=8
  approach="PostTrainingStatic"
  alpha=0.5
  script="run_generation.py"
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

    if [ "${topology}" = "starcoder_3b" ]; then
        model_name_or_path="/tf_dataset2/models/pytorch/starcode_3b"
        alpha=0.5
    elif [ "${topology}" = "codellama_7b" ]; then
        model_name_or_path="codellama/CodeLlama-7b-hf"
        alpha=0.5
    fi

    if [ ${script} = "run_generation.py" ];then
        python ./${script} \
            --model ${model_name_or_path} \
            --output_dir ${tuned_checkpoint} \
            --dataset ${DATASET_NAME} \
            --sq \
            --alpha ${alpha} \
            ${extra_cmd}
    else
        echo "Error: Please provide the correct script."
        exit 1
    fi
}

main "$@"
