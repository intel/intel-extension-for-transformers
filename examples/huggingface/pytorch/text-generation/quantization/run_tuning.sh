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
  script="run_generation.py"
  alpha=0.5
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
    if [ "${topology}" = "gpt_j" ]; then
        alpha=1.0
        model_name_or_path="/tf_dataset2/models/pytorch/gpt-j-6B"
        extra_cmd=$extra_cmd" --sq --alpha ${alpha}"
        extra_cmd=$extra_cmd" --output_dir ${tuned_checkpoint}"
    elif [ "${topology}" = "gpt_j_woq_rtn" ]; then
        model_name_or_path="/tf_dataset2/models/pytorch/gpt-j-6B"
        extra_cmd=$extra_cmd" --woq"
    elif [ "${topology}" = "gpt_j_woq_bab" ]; then
        model_name_or_path="/tf_dataset2/models/pytorch/gpt-j-6B"
        extra_cmd=$extra_cmd" --bitsandbytes"
    elif [ "${topology}" = "gpt_j_woq_load4bit" ]; then
        model_name_or_path="/tf_dataset2/models/pytorch/gpt-j-6B"
        extra_cmd=$extra_cmd" --load_in_4bit True"
    elif [ "${topology}" = "gpt_j_woq_load8bit" ]; then
        model_name_or_path="/tf_dataset2/models/pytorch/gpt-j-6B"
        extra_cmd=$extra_cmd" --load_in_8bit True"
    elif [ "${topology}" = "gpt_j_mp" ]; then
        model_name_or_path="/tf_dataset2/models/pytorch/gpt-j-6B"
        extra_cmd=$extra_cmd" --mixed_precision"
    elif [ "${topology}" = "opt_1.3b" ]; then
        alpha=0.8
        model_name_or_path="facebook/opt-1.3b"
        extra_cmd=$extra_cmd" --sq --alpha ${alpha}"
        extra_cmd=$extra_cmd" --output_dir ${tuned_checkpoint}"
    elif [ "${topology}" = "opt_2.7b" ]; then
        alpha=0.8
        model_name_or_path="facebook/opt-2.7b"
        extra_cmd=$extra_cmd" --sq --alpha ${alpha}"
        extra_cmd=$extra_cmd" --output_dir ${tuned_checkpoint}"
    elif [ "${topology}" = "opt_6.7b" ]; then
        alpha=0.8
        model_name_or_path="facebook/opt-6.7b"
        extra_cmd=$extra_cmd" --sq --alpha ${alpha}"
        extra_cmd=$extra_cmd" --output_dir ${tuned_checkpoint}"
    elif [ "${topology}" = "bloom_7b1" ]; then
        model_name_or_path="/tf_dataset2/models/pytorch/bloom-7b1"
        extra_cmd=$extra_cmd" --sq --alpha ${alpha}"
        extra_cmd=$extra_cmd" --output_dir ${tuned_checkpoint}"
    elif [ "${topology}" = "bloom_1b7" ]; then
        model_name_or_path="/tf_dataset2/models/pytorch/bloom-1b7"
        extra_cmd=$extra_cmd" --sq --alpha ${alpha}"
        extra_cmd=$extra_cmd" --output_dir ${tuned_checkpoint}"
    elif [ "${topology}" = "bloomz-3b" ]; then
        model_name_or_path="bigscience/bloomz-3b"
        extra_cmd=$extra_cmd" --sq --alpha ${alpha}"
        extra_cmd=$extra_cmd" --output_dir ${tuned_checkpoint}"
    elif [ "${topology}" = "llama_7b" ]; then
        alpha=0.7
        model_name_or_path="/tf_dataset2/models/pytorch/llama_7b"
        extra_cmd=$extra_cmd" --sq --alpha ${alpha}"
        extra_cmd=$extra_cmd" --output_dir ${tuned_checkpoint}"
    elif [ "${topology}" = "llama_13b" ]; then
        alpha=0.8
        model_name_or_path="decapoda-research/llama-13b-hf"
        extra_cmd=$extra_cmd" --sq --alpha ${alpha}"
        extra_cmd=$extra_cmd" --output_dir ${tuned_checkpoint}"
    elif [ "${topology}" = "dolly_v2_3b" ]; then
        alpha=0.6
        model_name_or_path="/tf_dataset2/models/pytorch/dolly_v2_3b"
        extra_cmd=$extra_cmd" --sq --alpha ${alpha}"
        extra_cmd=$extra_cmd" --output_dir ${tuned_checkpoint}"
    elif [ "${topology}" = "mpt_7b_chat" ]; then
        alpha=1.0
        model_name_or_path="mosaicml/mpt-7b-chat"
        extra_cmd=$extra_cmd" --sq --alpha ${alpha}"
        extra_cmd=$extra_cmd" --output_dir ${tuned_checkpoint}"
    fi

    if [ ${script} = "run_generation.py" ];then
        python -u ./${script} \
            --model ${model_name_or_path} \
            ${extra_cmd}
    else
        echo "Error: Please provide the correct script."
        exit 1
    fi
}

main "$@"
