#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {
  topology="gpt_j"
  tuned_checkpoint="saved_results"
  DATASET_NAME="NeelNanda/pile-10k"
  model_name_or_path="EleutherAI/gpt-j-6b"
  extra_cmd=""
  batch_size=8
  approach="PostTrainingStatic"
  script="vllm_acceleration_example.py"
  alpha=0.5
  weight_dtype="int4_clip"
  scheme="asym"
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
       --weight_dtype=*)
           weight_dtype=$(echo $var |cut -f2 -d=)
       ;;
       --bits=*)
           bits=$(echo $var |cut -f2 -d=)
       ;;
       --scheme=*)
           scheme=$(echo $var |cut -f2 -d=)
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
    if [ "${topology}" = "chatglm2_6b" ]; then
        model_name_or_path="THUDM/chatglm2-6b"
        script="vllm_acceleration_example.py"
    fi

    if [ ${script} = "vllm_acceleration_example.py" ];then
        pip install requirement.txt
        python -u ./${script} \
            --model ${model_name_or_path} \
            --prompt=你好
    else
        echo "Error: Please provide the correct script."
        exit 1
    fi
}

main "$@"
