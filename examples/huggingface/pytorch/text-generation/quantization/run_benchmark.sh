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
        echo "Error: Only support benchmark now."
        echo "Please go to language modeling folder to get accuracy."
        exit 1
    elif [[ ${mode} == "benchmark" ]]; then
        mode_cmd=" --benchmark "
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi


    if [ "${topology}" = "gpt_j" ]; then
        if [ "${task}" = "generation" ]; then
            script="run_generation.py"
        fi
        DATASET_NAME="NeelNanda/pile-10k"
        model_name_or_path="/tf_dataset2/models/pytorch/gpt-j-6B"
        if [ "${backend}" = "ipex" ]; then
            extra_cmd=$extra_cmd" --ipex"
        fi
    # elif [ "${topology}" = "opt_2.7b" ]; then
    #     script="run_generation.py"
    #     model_name_or_path="facebook/opt-2.7b"
    #     if [ "${backend}" = "ipex" ]; then
    #         extra_cmd=$extra_cmd" --ipex"
    #     fi
    # elif [ "${topology}" = "opt_6.7b" ]; then
    #     script="run_generation.py"
    #     model_name_or_path="facebook/opt-6.7b"
    #     if [ "${backend}" = "ipex" ]; then
    #         extra_cmd=$extra_cmd" --ipex"
    #     fi
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
    elif [ "${topology}" = "bloom_7b1" ]; then
        script="run_generation.py"
        # model_name_or_path="bigscience/bloom-7b1"
        model_name_or_path="/tf_dataset2/models/pytorch/bloom-7b1"
        if [ "${backend}" = "ipex" ]; then
            extra_cmd=$extra_cmd" --ipex"
        fi
    elif [ "${topology}" = "bloomz-3b" ]; then
        script="run_generation.py"
        model_name_or_path="bigscience/bloomz-3b"
        if [ "${backend}" = "ipex" ]; then
            extra_cmd=$extra_cmd" --ipex"
        fi
    elif [ "${topology}" = "dolly_v2_3b" ]; then
        script="run_generation.py"
        model_name_or_path="/tf_dataset2/models/pytorch/dolly_v2_3b"
        if [ "${backend}" = "ipex" ]; then
            extra_cmd=$extra_cmd" --ipex"
        fi
    fi

    
    if [[ ${int8} == "true" ]]; then
        extra_cmd=$extra_cmd" --int8"
    fi

    echo $extra_cmd

    if [ "${script}" == "run_generation.py" ];then
        python -u ./${script} \
            --model ${model_name_or_path} \
            --benchmark \
            --output_dir ${tuned_checkpoint} \
            --batch_size ${batch_size} \
            ${mode_cmd} \
            ${extra_cmd}
    else
        echo "Error: Please provide the correct script."
    fi
}

main "$@"
