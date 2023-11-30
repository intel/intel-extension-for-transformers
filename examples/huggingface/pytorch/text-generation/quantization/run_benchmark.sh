#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  iters=100
  batch_size=1
  tuned_checkpoint=saved_results
  lm_eval_tasks="lambada_openai "
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


# run_benchmark
function run_benchmark {
    extra_cmd=''

    if [[ ${mode} == "accuracy" ]]; then
        mode_cmd=" --accuracy "
        extra_cmd=$extra_cmd" --tasks ${lm_eval_tasks}"
    elif [[ ${mode} == "benchmark" ]]; then
        mode_cmd=" --benchmark "
        extra_cmd=$extra_cmd" --iters ${iters}"
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi


    if [ "${topology}" = "gpt_j" ]; then
        model_name_or_path="/tf_dataset2/models/pytorch/gpt-j-6B"
    elif [ "${topology}" = "gpt_j_woq_rtn" ]; then
        model_name_or_path="/tf_dataset2/models/pytorch/gpt-j-6B"
    elif [ "${topology}" = "gpt_j_woq_bab" ]; then
        model_name_or_path="/tf_dataset2/models/pytorch/gpt-j-6B"
    elif [ "${topology}" = "gpt_j_mp" ]; then
        model_name_or_path="/tf_dataset2/models/pytorch/gpt-j-6B"
    elif [ "${topology}" = "gpt_j_woq_load4bit" ]; then
	model_name_or_path="/tf_dataset2/models/pytorch/gpt-j-6B"
    elif [ "${topology}" = "gpt_j_woq_load8bit" ]; then
        model_name_or_path="/tf_dataset2/models/pytorch/gpt-j-6B"
    elif [ "${topology}" = "opt_1.3b" ]; then
        model_name_or_path="facebook/opt-1.3b"
    elif [ "${topology}" = "opt_2.7b" ]; then
        model_name_or_path="facebook/opt-2.7b"
    elif [ "${topology}" = "opt_6.7b" ]; then
        model_name_or_path="facebook/opt-6.7b"
    elif [ "${topology}" = "bloom_7b1" ]; then
        model_name_or_path="/tf_dataset2/models/pytorch/bloom-7b1"
    elif [ "${topology}" = "bloom_1b7" ]; then
        model_name_or_path="/tf_dataset2/models/pytorch/bloom-1b7"
    elif [ "${topology}" = "bloomz-3b" ]; then
        model_name_or_path="bigscience/bloomz-3b"
    elif [ "${topology}" = "llama_7b" ]; then
        model_name_or_path="meta-llama/Llama-2-7b-chat-hf"
    elif [ "${topology}" = "llama_13b" ]; then
        model_name_or_path="meta-llama/Llama-2-13b-chat-hf"
    elif [ "${topology}" = "dolly_v2_3b" ]; then
        model_name_or_path="/tf_dataset2/models/pytorch/dolly_v2_3b"
    elif [ "${topology}" = "mpt_7b_chat" ]; then
        model_name_or_path="mosaicml/mpt-7b-chat"
    fi

    
    if [[ ${int8} == "true" ]]; then
        if [ "${topology}" = "gpt_j_woq_rtn" ]; then
            extra_cmd=$extra_cmd" --woq"
        elif [ "${topology}" = "gpt_j_woq_bab" ]; then
            extra_cmd=$extra_cmd" --bitsandbytes"
        elif [ "${topology}" = "gpt_j_woq_load4bit" ]; then
            extra_cmd=$extra_cmd" --load_in_4bit True"
        elif [ "${topology}" = "gpt_j_woq_load8bit" ]; then
            extra_cmd=$extra_cmd" --load_in_8bit True"
        elif [ "${topology}" = "gpt_j_mp" ]; then
            extra_cmd=$extra_cmd" --mixed_precision"
        else
            extra_cmd=$extra_cmd" --int8"
        fi
    fi

    echo $extra_cmd

    if [ "${script}" == "run_generation.py" ];then
        python -u ./${script} \
            --model ${model_name_or_path} \
            --output_dir ${tuned_checkpoint} \
            --batch_size ${batch_size} \
            ${mode_cmd} \
            ${extra_cmd}
    else
        echo "Error: Please provide the correct script."
    fi
}

main "$@"
