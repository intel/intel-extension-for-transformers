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
  DATASET_NAME="wikitext"
  model_name_or_path="EleutherAI/gpt-neo-125m"
  extra_cmd=""
  batch_size=8
  model_type="bert"
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
    if [ "${topology}" = "" ]; then
        script="run_clm.py"
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="EleutherAI/gpt-neo-125m"
        task="clm"
        approach="PostTrainingStatic"
        backend=""
    elif [ "${topology}" = "gpt_neo" ]; then
        if [ "${task}" = "clm" ]; then
            script="run_clm.py"
        fi
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="EleutherAI/gpt-neo-125M"
        if [ "${approach}" = "dynamic" ]; then
            approach="PostTrainingDynamic"
        elif [ "${approach}" = "static" ]; then
            approach="PostTrainingStatic"
        elif [ "${approach}" = "qat" ]; then
            approach="QuantizationAwareTraining"
            extra_cmd=$extra_cmd" --learning_rate 1e-5 \
                    --num_train_epochs 6 \
                    --eval_steps 100 \
                    --save_steps 100 \
                    --greater_is_better True \
                    --load_best_model_at_end True \
                    --evaluation_strategy steps \
                    --save_strategy steps \
                    --save_total_limit 1"
        fi
    elif [ "${topology}" = "gpt_j" ]; then
        if [ "${task}" = "clm" ]; then
            script="run_clm.py"
        fi
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="/tf_dataset2/models/pytorch/gpt-j-6B"
        if [ "${approach}" = "dynamic" ]; then
            approach="PostTrainingDynamic"
        elif [ "${approach}" = "static" ]; then
            approach="PostTrainingStatic"
        fi
    elif [ "${topology}" = "bert" ]; then
        if [ "${task}" = "mlm" ]; then
            script="run_mlm.py"
        fi
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="bert-base-uncased"
        if [ "${approach}" = "dynamic" ]; then
            approach="PostTrainingDynamic"
        elif [ "${approach}" = "static" ]; then
            approach="PostTrainingStatic"
        elif [ "${approach}" = "qat" ]; then
            approach="QuantizationAwareTraining"
            extra_cmd=$extra_cmd" --learning_rate 1e-5 \
                    --num_train_epochs 6 \
                    --eval_steps 100 \
                    --save_steps 100 \
                    --greater_is_better True \
                    --load_best_model_at_end True \
                    --evaluation_strategy steps \
                    --save_strategy steps \
                    --metric_for_best_model accuracy \
                    --save_total_limit 1"
        fi
    elif [ "${topology}" = "xlnet" ]; then
        if [ "${task}" = "plm" ]; then
            script="run_plm.py"
        fi
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="xlnet-base-cased"
        if [ "${approach}" = "dynamic" ]; then
            approach="PostTrainingDynamic"
        elif [ "${approach}" = "static" ]; then
            approach="PostTrainingStatic"
        elif [ "${approach}" = "qat" ]; then
            approach="QuantizationAwareTraining"
            extra_cmd=$extra_cmd" --learning_rate 1e-5 \
                    --num_train_epochs 6 \
                    --eval_steps 100 \
                    --save_steps 100 \
                    --greater_is_better True \
                    --load_best_model_at_end True \
                    --evaluation_strategy steps \
                    --save_strategy steps \
                    --metric_for_best_model accuracy \
                    --save_total_limit 1"
        fi
    elif [ "${topology}" = "gpt_neox" ]; then
        if [ "${task}" = "clm" ]; then
            script="run_clm.py"
        fi
        DATASET_NAME="oscar"
        DATASET_CONFIG_NAME="unshuffled_original_ast"
        model_name_or_path="abeja/gpt-neox-japanese-2.7b"
        if [ "${approach}" = "dynamic" ]; then
            approach="PostTrainingDynamic"
        elif [ "${approach}" = "static" ]; then
            approach="PostTrainingStatic"
        fi
    elif [ "${topology}" = "bloom" ]; then
        if [ "${task}" = "clm" ]; then
            script="run_clm.py"
        fi
        DATASET_NAME="lambada"
        model_name_or_path="bigscience/bloom-560m"
        if [ "${approach}" = "static" ]; then
            approach="PostTrainingStatic"
        fi
        extra_cmd=$extra_cmd" --smooth_quant --sampling_size 400 --torchscript"
    fi

    if [ -z ${DATASET_CONFIG_NAME} ];then
        python -u ./${script} \
            --model_name_or_path ${model_name_or_path} \
            --dataset_name ${DATASET_NAME} \
            --do_eval \
            --do_train \
            --per_device_eval_batch_size ${batch_size} \
            --output_dir ${tuned_checkpoint} \
            --no_cuda \
            --tune \
            --overwrite_output_dir \
            --overwrite_cache \
            --quantization_approach ${approach} \
            ${extra_cmd}
    else
        python -u ./${script} \
            --model_name_or_path ${model_name_or_path} \
            --dataset_name ${DATASET_NAME} \
            --dataset_config_name ${DATASET_CONFIG_NAME} \
            --do_eval \
            --do_train \
            --per_device_eval_batch_size ${batch_size} \
            --output_dir ${tuned_checkpoint} \
            --no_cuda \
            --tune \
            --overwrite_output_dir \
            --overwrite_cache \
            --quantization_approach ${approach} \
            ${extra_cmd}
    fi
}

main "$@"
