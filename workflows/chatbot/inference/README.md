Chatbot Inference
============

This document shows how to perform the chatbot inference. You may refer to the [fine-tuning](../fine_tuning/README.md) section to understand how to fine-tune the model for chat. The chatbot inference has been validated on the 4th Gen Intel® Xeon® Processors, Sapphire Rapids (SPR) and Habana® Gaudi® Deep Learning Processors.

# Inference on Xeon SPR

## Setup Environment
To facilitate the process of setting up the inference environment, we have prepared a Dockerfile. This Dockerfile contains all the necessary configurations and dependencies required for running inference on the Xeon SPR platform. The provided [README guide](./docker/README.md) will walk you through the steps to build the Docker image and launch the Docker container for inference.

## MPT BF16 Inference

We provide the [generate.py](./generate.py) script for performing inference on Intel® CPUs. We have enabled IPEX BF16 to speed up the inference. Please use the following commands for inference.

For [MPT](https://huggingface.co/mosaicml/mpt-7b-chat), it uses the gpt-neox-20b tokenizer, so you need to explicitly define it in the command line.

If you don't have a fine-tuned model, please remove the 'peft_model_path' parameter.

```bash
# chat task
# recommended settings (e.g., -m 0 -C 0-55)
numactl -m <node N> -C <cpu list> python generate.py \
        --base_model_path "mosaicml/mpt-7b-chat" \
        --tokenizer_name "EleutherAI/gpt-neox-20b" \
        --use_kv_cache \
        --task chat \
        --instructions "Transform the following sentence into one that shows contrast. The tree is rotten." \
        --jit
```


To enable FP32 inference, you can add the parameter `--dtype "float32"`. To check the statistical information of inference, you can add the parameter `--return_stats`.

## LLama2 BF16 Inference
For Llama2, use the below command line to chat with it.
If you encounter a failure with the Llama fast tokenizer while using the latest transformers, add the option "--use_slow_tokenizer".
The `tokenizer_class` in `tokenizer_config.json` should be changed from `LLaMATokenizer` to `LlamaTokenizer`.
The `architectures` in `config.json` should be changed from `LLaMAForCausalLM` to `LlamaForCausalLM`.

```bash
# recommended settings (e.g., -m 0 -C 0-55)
numactl -m <node N> -C <cpu list> python generate.py \
        --base_model_path "meta-llama/Llama-2-7b-chat-hf" \
        --use_kv_cache \
        --task chat \
        --instructions "Transform the following sentence into one that shows contrast. The tree is rotten."
```

To enable FP32 inference, you can add the parameter `--dtype "float32"`. To check the statistical information of inference, you can add the parameter `--return_stats`.

## LLama2 INT8 Inference
[Llama2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) int8 inference demonstrates in [int8_llama2](https://github.com/intel/intel-extension-for-transformers/tree/int8_llama2/workflows/chatbot/inference) branch and need install Intel-extension-for-pytorch [llm_feature_branch](https://github.com/intel/intel-extension-for-pytorch/tree/llm_feature_branch) branch. Please follow the [README.md](https://github.com/intel/intel-extension-for-transformers/blob/81a4484dcc93f09d7609e6896fe3fbc22756975b/workflows/chatbot/inference/README.md) to setup the environments and make quantization.

# Inference on Habana Gaudi

Use this [link](https://docs.habana.ai/en/latest/AWS_EC2_DL1_and_PyTorch_Quick_Start/AWS_EC2_DL1_and_PyTorch_Quick_Start.html) to get started with setting up Gaudi-based Amazon EC2 DL1 instances.

## Setup Environment

```bash
git clone https://github.com/intel/intel-extension-for-transformers.git
cd ./intel-extension-for-transformers/
```

Copy the [generate.py](./generate.py) script to Gaudi instance and place it in the current directory.
Run the Docker container with Habana runtime and necessary environment variables:

```bash
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host -v $(pwd):/intel-extension-for-transformers vault.habana.ai/gaudi-docker/1.12.0/ubuntu22.04/habanalabs/pytorch-installer-2.0.1:latest
apt-get update
apt-get install git-lfs
git-lfs install
cd /intel-extension-for-transformers/workflows/chatbot/inference/
pip install datasets
pip install optimum
pip install git+https://github.com/huggingface/optimum-habana.git
pip install peft
pip install einops
pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.12.0
```

## Run the inference

You can use the [generate.py](./generate.py) script for performing direct inference on Habana Gaudi instance. We have enabled BF16 to speed up the inference. Please use the following command for inference.

```bash
python generate.py --base_model_path "mosaicml/mpt-7b-chat" \
             --habana \
             --tokenizer_name "EleutherAI/gpt-neox-20b" \
             --use_hpu_graphs \
             --use_kv_cache \
             --task chat \
             --instructions "Transform the following sentence into one that shows contrast. The tree is rotten."
```

And you can use `deepspeed` to run the large model.

```bash
python ../utils/gaudi_spawn.py --use_deepspeed --world_size 8 generate.py \
        --base_model_path "meta-llama/Llama-2-70b-chat-hf" \
        --habana \
        --use_hpu_graphs \
        --use_kv_cache \
        --task chat \
        --instructions "Transform the following sentence into one that shows contrast. The tree is rotten."
```

Habana supports HPU graph mode for inference speedup, which is available for bloom, gpt2, opt, gptj, gpt_neox, mpt, llama. You can use the parameter `use_hpu_graphs` to speed up the inference.

you can use '--peft_model_path' to apply you peft finetuned output model during generation.

```bash
python ../utils/gaudi_spawn.py --use_deepspeed --world_size 8 generate.py \
        --base_model_path "meta-llama/Llama-2-70b-chat-hf" \
        --peft_model_path <peft_model_output_folder>
        --habana \
        --use_hpu_graphs \
        --use_kv_cache \
        --task chat \
        --instructions "Transform the following sentence into one that shows contrast. The tree is rotten."
```

# Additional Notes

Here are the explanations of parameters in generate.py:
`--temperature`: Controls the diversity of generated text. Lower values result in more deterministic outputs. The default value is 0.1.
`--top_p`: During text generation, only consider tokens with cumulative probability up to this value. This parameter helps to avoid extremely low probability tokens. The default value is 0.75.
`--top_k`: The number of highest probability vocabulary tokens to consider for each step of text generation. The default value is 40.
`--num_beams`: The number of beams to use for beam search decoding. This parameter helps to generate multiple possible completions. The default value is 1.
`--repetition_penalty`: This value controls the penalty for repeating tokens in the generated text. Higher values encourage the model to produce more diverse outputs. The default value is 1.1.
`--max_new_tokens`: The maximum number of tokens allowed in the generated output. This parameter helps to limit the length of the generated text. The default value is 128.
