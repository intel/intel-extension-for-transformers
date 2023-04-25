#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  iters=100
  batch_size=16
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
      --bf16=*)
          int8=$(echo ${var} |cut -f2 -d=)
      ;;
      --config=*)
          tuned_checkpoint=$(echo $var |cut -f2 -d=)
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
        mode_cmd=" --accuracy_only"
    elif [[ ${mode} == "benchmark" ]]; then
        mode_cmd=" --benchmark "
    elif [[ ${mode} == "benchmark_only" ]]; then
        mode_cmd=" --benchmark_only "
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi

    if [ "${topology}" = "gpt_neo_clm_static" ]; then
        script="run_clm.py"
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="EleutherAI/gpt-neo-125M"
    elif [ "${topology}" = "gpt_neo_clm_dynamic" ]; then
        script="run_clm.py"
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="EleutherAI/gpt-neo-125M"
    elif [ "${topology}" = "gptj_clm_static" ]; then
        script="run_clm.py"
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="/tf_dataset2/models/pytorch/gpt-j-6B"
    elif [ "${topology}" = "gptj_clm_dynamic" ]; then
        script="run_clm.py"
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="/tf_dataset2/models/pytorch/gpt-j-6B"
    elif [ "${topology}" = "gpt_j_6b_clm_ipex" ]; then
        script="evaluate_clm.py"
        model_name_or_path="/tf_dataset2/models/pytorch/gpt-j-6B"
        approach="PostTrainingStatic"
    elif [ "${topology}" = "bert_mlm_static" ]; then
        script="run_mlm.py"
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="bert-base-uncased"
    elif [ "${topology}" = "bert_mlm_dynamic" ]; then
        script="run_mlm.py"
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="bert-base-uncased"
    elif [ "${topology}" = "xlnet_plm_static" ]; then
        script="run_plm.py"
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="xlnet-base-cased"
    elif [ "${topology}" = "xlnet_plm_dynamic" ]; then
        script="run_plm.py"
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="xlnet-base-cased"
    elif [ "${topology}" = "reformer_crime_and_punishment_static" ]; then
        script="run_clm.py"
        DATASET_NAME="crime_and_punish"
        model_name_or_path="google/reformer-crime-and-punishment"
    elif [ "${topology}" = "ctrl_wikitext_static" ]; then
        script="run_clm.py"
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="sshleifer/tiny-ctrl"
    elif [ "${topology}" = "gpt_neox_clm_static" ]; then
        script="run_clm.py"
        DATASET_NAME="oscar"
        DATASET_CONFIG_NAME="unshuffled_original_ast"
        model_name_or_path="abeja/gpt-neox-japanese-2.7b"
    elif [ "${topology}" = "gpt_neox_clm_dynamic" ]; then
        script="run_clm.py"
        DATASET_NAME="oscar"
        DATASET_CONFIG_NAME="unshuffled_original_ast"
        model_name_or_path="abeja/gpt-neox-japanese-2.7b"
    elif [ "${topology}" = "bloom_clm_static" ]; then
        script="run_clm.py"
        DATASET_NAME="lambada"
        model_name_or_path="bigscience/bloom-560m"
    elif [ "${topology}" = "gpt_neo_clm_qat" ]; then
        script="run_clm.py"
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="EleutherAI/gpt-neo-125M"
    elif [ "${topology}" = "bert_mlm_qat" ]; then
        script="run_mlm.py"
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="bert-base-uncased"
    elif [ "${topology}" = "xlnet_plm_qat" ]; then
        script="run_plm.py"
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="xlnet-base-cased"
    fi
    
    if [[ ${int8} == "true" ]]; then
        extra_cmd=$extra_cmd" --int8"
        model_name_or_path=${tuned_checkpoint}
    fi

    if [[ ${int8} == "true" ]] && [ "${topology}" = "gpt_j_6b_clm_ipex" ]; then
        model_name_or_path="/tf_dataset2/models/pytorch/gpt-j-6B"
    fi

    if [[ ${bf16} == "true" ]]; then
        extra_cmd=$extra_cmd" --bf16_ipex"
    fi

    echo $extra_cmd

    if [ ${script} = "evaluate_clm.py" ];then
        python -u ./${script} \
            --model ${model_name_or_path} \
            --output_dir ${tuned_checkpoint} \
	    --batch_size ${batch_size} \
            ${mode_cmd} \
            ${extra_cmd}
    elif [ -z ${DATASET_CONFIG_NAME} ];then
        python -u ${script} \
            --model_name_or_path ${model_name_or_path} \
            --dataset_name ${DATASET_NAME} \
            --do_eval \
            --per_device_eval_batch_size ${batch_size} \
            --output_dir ./tmp/benchmark_output \
            --overwrite_output_dir \
            --overwrite_cache \
            --no_cuda \
            ${mode_cmd} \
            ${extra_cmd}
    else
        python -u ${script} \
            --model_name_or_path ${model_name_or_path} \
            --dataset_name ${DATASET_NAME} \
            --dataset_config_name ${DATASET_CONFIG_NAME} \
            --do_eval \
            --per_device_eval_batch_size ${batch_size} \
            --output_dir ./tmp/benchmark_output \
            --overwrite_output_dir \
            --overwrite_cache \
            --no_cuda \
            ${mode_cmd} \
            ${extra_cmd}
    fi
}

main "$@"
