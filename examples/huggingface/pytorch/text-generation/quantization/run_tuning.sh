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
  DATASET_NAME="NeelNanda/pile-10k"
  model_name_or_path="EleutherAI/gpt-j-6b"
  extra_cmd=""
  batch_size=8
  approach="PostTrainingStatic"
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
        if [ "${task}" = "generation" ]; then
            script="run_generation.py"
        fi
        model_name_or_path="/tf_dataset2/models/pytorch/gpt-j-6B"
        if [ "${backend}" = "ipex" ]; then
            extra_cmd=$extra_cmd" --ipex"
            extra_cmd=$extra_cmd" --int8_bf16_mixed"
            alpha=1.0
        fi
    elif [ "${topology}" = "opt_1.3b" ]; then
        script="run_generation.py"
        model_name_or_path="facebook/opt-1.3b"
        if [ "${backend}" = "ipex" ]; then
           extra_cmd=$extra_cmd" --ipex"
           alpha=0.8
        fi
    elif [ "${topology}" = "opt_2.7b" ]; then
        script="run_generation.py"
        model_name_or_path="facebook/opt-2.7b"
        if [ "${backend}" = "ipex" ]; then
           extra_cmd=$extra_cmd" --ipex"
	   alpha=0.8
        fi
    elif [ "${topology}" = "opt_6.7b" ]; then
        script="run_generation.py"
        model_name_or_path="facebook/opt-6.7b"
        if [ "${backend}" = "ipex" ]; then
            extra_cmd=$extra_cmd" --ipex"
	    alpha=0.8
        fi
    elif [ "${topology}" = "bloom_7b1" ]; then
        script="run_generation.py"
        model_name_or_path="/tf_dataset2/models/pytorch/bloom-7b1"
        if [ "${backend}" = "ipex" ]; then
            extra_cmd=$extra_cmd" --ipex"
        fi
    elif [ "${topology}" = "bloom_1b7" ]; then
        script="run_generation.py"
        model_name_or_path="/tf_dataset2/models/pytorch/bloom-1b7"
        if [ "${backend}" = "ipex" ]; then
            extra_cmd=$extra_cmd" --ipex"
        fi
    elif [ "${topology}" = "bloomz-3b" ]; then
        script="run_generation.py"
        model_name_or_path="bigscience/bloomz-3b"
        if [ "${backend}" = "ipex" ]; then
            extra_cmd=$extra_cmd" --ipex"
        fi
    elif [ "${topology}" = "llama_7b" ]; then
        script="run_generation.py"
        model_name_or_path="/tf_dataset2/models/pytorch/llama_7b"
        if [ "${backend}" = "ipex" ]; then
            extra_cmd=$extra_cmd" --ipex"
        fi
    elif [ "${topology}" = "llama_13b" ]; then
        script="run_generation.py"
        model_name_or_path="decapoda-research/llama-13b-hf"
        if [ "${backend}" = "ipex" ]; then
            extra_cmd=$extra_cmd" --ipex"
        fi
    elif [ "${topology}" = "dolly_v2_3b" ]; then
        script="run_generation.py"
        model_name_or_path="/tf_dataset2/models/pytorch/dolly_v2_3b"
        if [ "${backend}" = "ipex" ]; then
            extra_cmd=$extra_cmd" --ipex"
	    extra_cmd=$extra_cmd" --int8_bf16_mixed"
	    alpha=1.0
        fi
    elif [ "${topology}" = "mpt_7b_chat" ]; then
        script="run_generation.py"
        model_name_or_path="mosaicml/mpt-7b-chat"
        if [ "${backend}" = "ipex" ]; then
            extra_cmd=$extra_cmd" --ipex"
            alpha=0.95
        fi
    fi

    if [ ${script} = "run_generation.py" ];then
        python -u ./${script} \
            --model ${model_name_or_path} \
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
