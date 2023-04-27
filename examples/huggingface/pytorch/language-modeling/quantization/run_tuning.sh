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
  model_name_or_path="EleutherAI/gpt-neo-125M"
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
      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;
    esac
  done

}

# run_tuning
function run_tuning {
    if [ "${topology}" = "gpt_neo_clm_static" ]; then
        script="run_clm.py"
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="EleutherAI/gpt-neo-125M"
        approach="PostTrainingStatic"
    elif [ "${topology}" = "gpt_neo_clm_dynamic" ]; then
        script="run_clm.py"
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="EleutherAI/gpt-neo-125M"
        approach="PostTrainingDynamic"
    elif [ "${topology}" = "gpt_neo_clm_qat" ]; then
        script="run_clm.py"
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="EleutherAI/gpt-neo-125M"
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
    elif [ "${topology}" = "gptj_clm_static" ]; then
        script="run_clm.py"
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="/tf_dataset2/models/pytorch/gpt-j-6B"
        approach="PostTrainingStatic"
    elif [ "${topology}" = "gptj_clm_dynamic" ]; then
        script="run_clm.py"
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="/tf_dataset2/models/pytorch/gpt-j-6B"
        approach="PostTrainingDynamic"
    elif [ "${topology}" = "gpt_j_6b_clm_ipex" ]; then
        script="evaluate_clm.py"
        DATASET_NAME="NeelNanda/pile-10k"
        model_name_or_path="/tf_dataset2/models/pytorch/gpt-j-6B"
        approach="PostTrainingStatic"
    elif [ "${topology}" = "opt_2.7b_clm_ipex" ]; then
        script="evaluate_clm.py"
        DATASET_NAME="NeelNanda/pile-10k"
        model_name_or_path="facebook/opt-2.7b"
        approach="PostTrainingStatic"
	extra_cmd=$extra_cmd" --int8_bf16_mixed"
    elif [ "${topology}" = "opt_6.7b_clm_ipex" ]; then
        script="evaluate_clm.py"
        DATASET_NAME="NeelNanda/pile-10k"
        model_name_or_path="facebook/opt-6.7b"
        approach="PostTrainingStatic"
	extra_cmd=$extra_cmd" --int8_bf16_mixed"
    elif [ "${topology}" = "llama_7b_clm_ipex" ]; then
        script="evaluate_clm.py"
        DATASET_NAME="NeelNanda/pile-10k"
        model_name_or_path="decapoda-research/llama-7b-hf"
        approach="PostTrainingStatic"
	extra_cmd=$extra_cmd" --int8_bf16_mixed"
    elif [ "${topology}" = "bert_mlm_static" ]; then
        script="run_mlm.py"
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="bert-base-uncased"
        approach="PostTrainingStatic"
    elif [ "${topology}" = "bert_mlm_qat" ]; then
        script="run_mlm.py"
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="bert-base-uncased"
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
    elif [ "${topology}" = "bert_mlm_dynamic" ]; then
        script="run_mlm.py"
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="bert-base-uncased"
        approach="PostTrainingDynamic"
    elif [ "${topology}" = "xlnet_plm_static" ]; then
        script="run_plm.py"
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="xlnet-base-cased"
        approach="PostTrainingStatic"
    elif [ "${topology}" = "xlnet_plm_qat" ]; then
        script="run_plm.py"
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="xlnet-base-cased"
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
    elif [ "${topology}" = "xlnet_plm_dynamic" ]; then
        script="run_plm.py"
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="xlnet-base-cased"
        approach="PostTrainingDynamic"
    elif [ "${topology}" = "reformer_crime_and_punishment_static" ]; then
        script="run_clm.py"
        DATASET_NAME="crime_and_punish"
        model_name_or_path="google/reformer-crime-and-punishment"
        approach="PostTrainingStatic"
    elif [ "${topology}" = "ctrl_wikitext_static" ]; then
        script="run_clm.py"
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="sshleifer/tiny-ctrl"
        approach="PostTrainingStatic"
    elif [ "${topology}" = "gpt_neox_clm_static" ]; then
        script="run_clm.py"
        DATASET_NAME="oscar"
        DATASET_CONFIG_NAME="unshuffled_original_ast"
        model_name_or_path="abeja/gpt-neox-japanese-2.7b"
        approach="PostTrainingStatic"
    elif [ "${topology}" = "gpt_neox_clm_dynamic" ]; then
        script="run_clm.py"
        DATASET_NAME="oscar"
        DATASET_CONFIG_NAME="unshuffled_original_ast"
        model_name_or_path="abeja/gpt-neox-japanese-2.7b"
        approach="PostTrainingDynamic"
    elif [ "${topology}" = "bloom_clm_static" ]; then
        script="run_clm.py"
        DATASET_NAME="lambada"
        model_name_or_path="bigscience/bloom-560m"
        approach="PostTrainingStatic"
        extra_cmd=$extra_cmd" --smooth_quant --sampling_size 400 --torchscript"
    fi

    if [ ${script} = "evaluate_clm.py" ];then
        python -u ./${script} \
            --model ${model_name_or_path} \
            --output_dir ${tuned_checkpoint} \
            --dataset ${DATASET_NAME} \
            --quantize \
            --sq \
            --alpha ${alpha} \
            ${extra_cmd}
    elif [ -z ${DATASET_CONFIG_NAME} ];then
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
