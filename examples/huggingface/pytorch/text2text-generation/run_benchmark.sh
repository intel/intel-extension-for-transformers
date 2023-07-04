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
  lm_eval_tasks="cnn_dailymail"
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
      --lm_eval_tasks=*)
          lm_eval_tasks=$(echo $var |cut -f2 -d=)
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
    script="run_seq2seq_generation.py"

    if [[ ${mode} == "accuracy" ]]; then
        mode_cmd=" --accuracy "
        extra_cmd=$extra_cmd" --tasks ${lm_eval_tasks}"
    elif [[ ${mode} == "benchmark" ]]; then
        mode_cmd=" --benchmark "
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi


    if [ "${topology}" = "flan-t5" ]; then
        DATASET_NAME="NeelNanda/pile-10k"
        model_type="t5"
        model_name_or_path="google/flan-t5-large"
        if [ "${backend}" = "ipex" ]; then
            extra_cmd=$extra_cmd" --ipex"
        fi
    fi

    
    if [[ ${int8} == "true" ]]; then
        extra_cmd=$extra_cmd" --int8"
    fi

    echo $extra_cmd

    if [ "${script}" == "run_seq2seq_generation.py" ];then
        python -u ./${script} \
            --model_type ${model_type} \
            --model_name_or_path ${model_name_or_path} \
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
