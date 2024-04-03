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
      --model_source=*)
          model_source=$(echo $var |cut -f2 -d=)
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
    elif [ "${topology}" = "llama2_7b_gptq" ]; then
        model_name_or_path="meta-llama/Llama-2-7b-hf"
    elif [ "${topology}" = "llama_13b" ]; then
        model_name_or_path="meta-llama/Llama-2-13b-chat-hf"
    elif [ "${topology}" = "dolly_v2_3b" ]; then
        model_name_or_path="/tf_dataset2/models/pytorch/dolly_v2_3b"
    elif [ "${topology}" = "mpt_7b_chat" ]; then
        model_name_or_path="mosaicml/mpt-7b-chat"
    elif [ "${topology}" = "chatglm3_6b" ]; then
        model_name_or_path="THUDM/chatglm3-6b"
        extra_cmd=$extra_cmd" --trust_remote_code"
    elif [ "${topology}" = "chatglm2_6b" ]; then
        model_name_or_path="THUDM/chatglm2-6b"
        extra_cmd=$extra_cmd" --trust_remote_code"
    elif [ "${topology}" = "chatglm_6b" ]; then
        model_name_or_path="THUDM/chatglm-6b"
        extra_cmd=$extra_cmd" --trust_remote_code"
        pip install transformers==4.33
    elif [ "${topology}" = "falcon_7b" ]; then
        model_name_or_path="tiiuae/falcon-7b-instruct"
        pip install transformers==4.33
    elif [ "${topology}" = "baichuan_7b" ]; then
        model_name_or_path="baichuan-inc/Baichuan-7B"
        extra_cmd=$extra_cmd" --trust_remote_code"
        pip install transformers==4.33
    elif [ "${topology}" = "baichuan_13b" ]; then
        model_name_or_path="baichuan-inc/Baichuan-13B-Base"
        extra_cmd=$extra_cmd" --trust_remote_code"
        extra_cmd=$extra_cmd" --_commit_hash 14d5b0e204542744900f6fb52422c6d633bdcb00"
        pip install transformers==4.33
    elif [ "${topology}" = "baichuan2_7b" ]; then
        model_name_or_path="baichuan-inc/Baichuan2-7B-Base"
        extra_cmd=$extra_cmd" --trust_remote_code"
        pip install transformers==4.33
    elif [ "${topology}" = "baichuan2_13b" ]; then
        model_name_or_path="baichuan-inc/Baichuan2-13B-Base"
        extra_cmd=$extra_cmd" --trust_remote_code"
        pip install transformers==4.35.2
    elif [ "${topology}" = "qwen_7b" ]; then
        model_name_or_path="Qwen/Qwen-7B"
        extra_cmd=$extra_cmd" --trust_remote_code"
        extra_cmd=$extra_cmd" --_commit_hash f7bc352f27bb1c02ee371a4576942a7d96c8bb97"
	      pip install transformers==4.35.2
    elif [ "${topology}" = "mistral_7b" ]; then
        model_name_or_path="Intel/neural-chat-7b-v3"
    elif [ "${topology}" = "phi_1b" ]; then
        model_name_or_path="susnato/phi-1_dev"
	    pip install transformers==4.36.1
    elif [ "${topology}" = "phi_1_5b" ]; then
        model_name_or_path="susnato/phi-1_5_dev"
	    pip install transformers==4.36.1
    elif [ "${topology}" = "llama2_7b_gptq" ] && [ "$model_source" != "huggingface" ]; then
        model_name_or_path="/tf_dataset2/models/nlp_toolkit/llama-2-7b-chat/Llama-2-7b-chat-hf"
    elif [ "${topology}" = "mistral_7b_autoround" ] && [ "$model_source" != "huggingface" ]; then
        model_name_or_path="/tf_dataset2/models/pytorch/Mistral-7B-v0.1"
    elif [ "${topology}" = "mistral_7b_rtn" ] && [ "$model_source" != "huggingface" ]; then
        model_name_or_path="/tf_dataset2/models/pytorch/Mistral-7B-v0.1"
    elif [ "${topology}" = "mistral_7b_gptq" ] && [ "$model_source" != "huggingface" ]; then
        model_name_or_path="/tf_dataset2/models/pytorch/Mistral-7B-v0.1"
    fi

    if [[ ${int8} == "true" ]]; then
        if [ "${topology}" = "gpt_j_woq_rtn" ]; then
            extra_cmd=$extra_cmd" --woq"
        elif [ "${topology}" = "gpt_j_woq_bab" ]; then
            extra_cmd=$extra_cmd" --bitsandbytes"
        elif [ "${topology}" = "gpt_j_woq_load4bit" ]; then
            extra_cmd=$extra_cmd" --load_in_4bit "
        elif [ "${topology}" = "gpt_j_woq_load8bit" ]; then
            extra_cmd=$extra_cmd" --load_in_8bit "
        elif [ "${topology}" = "gpt_j_mp" ]; then
            extra_cmd=$extra_cmd" --mixed_precision"
        elif [ "${topology}" = "llama2_7b_gptq" ]; then
            if [[ "$model_source" == "huggingface" ]]; then
                model_name_or_path="TheBloke/Llama-2-7B-Chat-GPTQ"
            else
                model_name_or_path="/tf_dataset2/models/nlp_toolkit/llama-2-7b-chat/Llama-2-7b-chat-hf"
                extra_cmd=$extra_cmd" --trust_remote_code"
                extra_cmd=$extra_cmd" --woq_loading"
            fi
        elif [ "${topology}" = "mistral_7b_autoround" ]; then
            if [[ "$model_source" == "huggingface" ]]; then
                model_name_or_path="Intel/Mistral-7B-v0.1-int4-inc"
            else
                model_name_or_path="/tf_dataset2/models/pytorch/Mistral-7B-v0.1"
                extra_cmd=$extra_cmd" --trust_remote_code"
                extra_cmd=$extra_cmd" --woq_loading"
            fi            
        elif [ "${topology}" = "mistral_7b_rtn" ]; then
            if [[ "$model_source" == "huggingface" ]]; then
                model_name_or_path="mistralai/Mistral-7B-v0.1"
            else
                model_name_or_path="/tf_dataset2/models/pytorch/Mistral-7B-v0.1"
                extra_cmd=$extra_cmd" --trust_remote_code"
                extra_cmd=$extra_cmd" --woq_loading"
            fi            
        elif [ "${topology}" = "mistral_7b_gptq" ]; then
            if [[ "$model_source" == "huggingface" ]]; then
                model_name_or_path="TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"
            else
                model_name_or_path="/tf_dataset2/models/pytorch/Mistral-7B-v0.1"
                extra_cmd=$extra_cmd" --trust_remote_code"
                extra_cmd=$extra_cmd" --woq_loading"
            fi
        else
            extra_cmd=$extra_cmd" --int8"
        fi
    fi
    if [[ $backend == "neuralspeed" ]]; then
        extra_cmd=$extra_cmd" --use_neural_speed"
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
