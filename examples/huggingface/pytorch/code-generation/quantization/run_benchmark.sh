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
  lm_eval_tasks="openai_humaneval"
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
	extra_cmd=$extra_cmd" --tasks ${lm_eval_tasks} --allow_code_execution --n_samples 20 --batch_size 20 --do_sample"
	batch_size=112
    elif [[ ${mode} == "benchmark" ]]; then
        mode_cmd=" --benchmark "
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi


    if [ "${topology}" = "starcoder_3b" ]; then
        model_name_or_path="/tf_dataset2/models/pytorch/starcode_3b"
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
            --output_dir ${tuned_checkpoint} \
            --batch_size ${batch_size} \
            ${mode_cmd} \
            ${extra_cmd}
    else
        echo "Error: Please provide the correct script."
    fi
}

main "$@"
