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
  weight_dtype="int4_clip"
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
        extra_cmd=$extra_cmd" --load_in_4bit"
    elif [ "${topology}" = "gpt_j_woq_load8bit" ]; then
        model_name_or_path="/tf_dataset2/models/pytorch/gpt-j-6B"
        extra_cmd=$extra_cmd" --load_in_8bit "
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
        model_name_or_path="/tf_dataset2/models/nlp_toolkit/llama-2-7b-chat/Llama-2-7b-chat-hf"
        extra_cmd=$extra_cmd" --sq --alpha ${alpha}"
        extra_cmd=$extra_cmd" --output_dir ${tuned_checkpoint}"
    elif [ "${topology}" = "llama_13b" ]; then
        alpha=0.8
        model_name_or_path="meta-llama/Llama-2-13b-chat-hf"
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
    elif [ "${topology}" = "chatglm3_6b" ]; then
        alpha=0.75
        model_name_or_path="THUDM/chatglm3-6b"
        extra_cmd=$extra_cmd" --sq --alpha ${alpha}"
        extra_cmd=$extra_cmd" --output_dir ${tuned_checkpoint}"
        extra_cmd=$extra_cmd" --trust_remote_code"
    elif [ "${topology}" = "chatglm2_6b" ]; then
        alpha=0.75
        model_name_or_path="THUDM/chatglm2-6b"
        extra_cmd=$extra_cmd" --sq --alpha ${alpha}"
        extra_cmd=$extra_cmd" --output_dir ${tuned_checkpoint}"
        extra_cmd=$extra_cmd" --trust_remote_code"
    elif [ "${topology}" = "chatglm_6b" ]; then
        alpha=0.75
        model_name_or_path="THUDM/chatglm-6b"
        extra_cmd=$extra_cmd" --sq --alpha ${alpha}"
        extra_cmd=$extra_cmd" --output_dir ${tuned_checkpoint}"
        extra_cmd=$extra_cmd" --trust_remote_code"
        pip install transformers==4.33
    elif [ "${topology}" = "falcon_7b" ]; then
        alpha=0.7
        model_name_or_path="tiiuae/falcon-7b-instruct"
        extra_cmd=$extra_cmd" --sq --alpha ${alpha}"
        extra_cmd=$extra_cmd" --output_dir ${tuned_checkpoint}"
	    pip install transformers==4.33.3
    elif [ "${topology}" = "baichuan_7b" ]; then
        alpha=0.85
        model_name_or_path="baichuan-inc/Baichuan-7B"
        extra_cmd=$extra_cmd" --sq --alpha ${alpha}"
        extra_cmd=$extra_cmd" --output_dir ${tuned_checkpoint}"
        extra_cmd=$extra_cmd" --trust_remote_code"
        pip install transformers==4.33
    elif [ "${topology}" = "baichuan_13b" ]; then
        alpha=0.85
        model_name_or_path="baichuan-inc/Baichuan-13B-Base"
        extra_cmd=$extra_cmd" --sq --alpha ${alpha}"
        extra_cmd=$extra_cmd" --output_dir ${tuned_checkpoint}"
        extra_cmd=$extra_cmd" --trust_remote_code"
        extra_cmd=$extra_cmd" --_commit_hash 14d5b0e204542744900f6fb52422c6d633bdcb00"
        pip install transformers==4.33
    elif [ "${topology}" = "baichuan2_7b" ]; then
        alpha=0.85
        model_name_or_path="baichuan-inc/Baichuan2-7B-Base"
        extra_cmd=$extra_cmd" --sq --alpha ${alpha}"
        extra_cmd=$extra_cmd" --output_dir ${tuned_checkpoint}"
        extra_cmd=$extra_cmd" --trust_remote_code"
        pip install transformers==4.33
    elif [ "${topology}" = "baichuan2_13b" ]; then
        alpha=0.55
        model_name_or_path="baichuan-inc/Baichuan2-13B-Base"
        extra_cmd=$extra_cmd" --sq --alpha ${alpha}"
        extra_cmd=$extra_cmd" --output_dir ${tuned_checkpoint}"
        extra_cmd=$extra_cmd" --trust_remote_code"
        pip install transformers==4.35.2
    elif [ "${topology}" = "qwen_7b" ]; then
        alpha=0.9
        model_name_or_path="Qwen/Qwen-7B"
        extra_cmd=$extra_cmd" --sq --alpha ${alpha}"
        extra_cmd=$extra_cmd" --output_dir ${tuned_checkpoint}"
        extra_cmd=$extra_cmd" --trust_remote_code"
        extra_cmd=$extra_cmd" --_commit_hash f7bc352f27bb1c02ee371a4576942a7d96c8bb97"
	      pip install transformers==4.35.2
    elif [ "${topology}" = "mistral_7b" ]; then
        alpha=0.8
        model_name_or_path="Intel/neural-chat-7b-v3"
        extra_cmd=$extra_cmd" --sq --alpha ${alpha}"
        extra_cmd=$extra_cmd" --output_dir ${tuned_checkpoint}"
        extra_cmd=$extra_cmd" --trust_remote_code"
    elif [ "${topology}" = "phi_1b" ]; then
        alpha=0.5
        model_name_or_path="susnato/phi-1_dev"
        extra_cmd=$extra_cmd" --sq --alpha ${alpha}"
        extra_cmd=$extra_cmd" --output_dir ${tuned_checkpoint}"
        extra_cmd=$extra_cmd" --trust_remote_code"
	    pip install transformers==4.36.1
    elif [ "${topology}" = "phi_1_5b" ]; then
        alpha=0.5
        model_name_or_path="susnato/phi-1_5_dev"
        extra_cmd=$extra_cmd" --sq --alpha ${alpha}"
        extra_cmd=$extra_cmd" --output_dir ${tuned_checkpoint}"
        extra_cmd=$extra_cmd" --trust_remote_code"
	    pip install transformers==4.36.1
    elif [ "${topology}" = "llama2_7b_gptq" ]; then
        model_name_or_path="meta-llama/Llama-2-7b-hf"
        extra_cmd=$extra_cmd" --woq --bits 4 --compute_dtype fp32 --scheme asym "
        extra_cmd=$extra_cmd" --woq_algo "GPTQ" --desc_act --blocksize 128 --max_input_length 2048 "
        extra_cmd=$extra_cmd" --output_dir ${tuned_checkpoint}"
        extra_cmd=$extra_cmd" --trust_remote_code"
        extra_cmd=$extra_cmd" --weight_dtype ${weight_dtype}"
    elif [ "${topology}" = "mistral_7b_autoround" ]; then
        model_name_or_path="/tf_dataset2/models/pytorch/Mistral-7B-v0.1"
        extra_cmd=$extra_cmd" --woq --bits 4 --compute_dtype fp32 --scheme asym "
        extra_cmd=$extra_cmd" --woq_algo "AutoRound" --desc_act --blocksize 128 --max_input_length 2048 "
        extra_cmd=$extra_cmd" --output_dir ${tuned_checkpoint}"
        extra_cmd=$extra_cmd" --trust_remote_code"
        extra_cmd=$extra_cmd" --weight_dtype ${weight_dtype}"
    elif [ "${topology}" = "mistral_7b_rtn" ]; then
        model_name_or_path="/tf_dataset2/models/pytorch/Mistral-7B-v0.1"
        extra_cmd=$extra_cmd" --woq --bits 4 --compute_dtype fp32 --scheme asym "
        extra_cmd=$extra_cmd" --woq_algo "Rtn" --desc_act --blocksize 128 --max_input_length 2048 "
        extra_cmd=$extra_cmd" --output_dir ${tuned_checkpoint}"
        extra_cmd=$extra_cmd" --trust_remote_code"
        extra_cmd=$extra_cmd" --weight_dtype ${weight_dtype}"
    elif [ "${topology}" = "mistral_7b_gptq" ]; then
        model_name_or_path="/tf_dataset2/models/pytorch/Mistral-7B-v0.1"
        extra_cmd=$extra_cmd" --woq --bits 4 --compute_dtype fp32 --scheme asym "
        extra_cmd=$extra_cmd" --woq_algo "GPTQ" --desc_act --blocksize 128 --max_input_length 2048 "
        extra_cmd=$extra_cmd" --output_dir ${tuned_checkpoint}"
        extra_cmd=$extra_cmd" --trust_remote_code"
        extra_cmd=$extra_cmd" --weight_dtype ${weight_dtype}"
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
