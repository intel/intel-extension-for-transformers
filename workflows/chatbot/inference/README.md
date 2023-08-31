Chatbot Inference
============

This document showcases the utilization of the fine-tuned model for conversing with NeuralChat. To obtaining the fine-tuned model, please refer to the [fine-tuning](../fine_tuning/README.md) section. The inference of the fine-tuned models has been validated on the 4th Gen Intel® Xeon® Processors, Sapphire Rapids(SPR) and Habana® Gaudi® Deep Learning Processors.

# Inference on Xeon SPR

## Setup Environment
TBD: @Liang enhance the docker file and use docker command here

## MPT BF16 Inference

We provide the [generate.py](./generate.py) script for performing inference on Intel® CPUs. We have enabled IPEX BF16 to speed up the inference. Please use the following commands for inference.

For [MPT](https://huggingface.co/mosaicml/mpt-7b-chat), it uses the gpt-neox-20b tokenizer, so you need to explicitly define it in the command line.

If you don't have a fine-tuned model, please remove the 'peft_model_path' parameter.

```bash
# chat task
python generate.py \
        --base_model_path "mosaicml/mpt-7b-chat" \
        --peft_model_path "./mpt_peft_finetuned_model" \
        --tokenizer_name "EleutherAI/gpt-neox-20b" \
        --use_kv_cache \
        --task chat \
        --instructions "Transform the following sentence into one that shows contrast. The tree is rotten." \
        --jit

```

TODO: @chang mention FP32 inference

## LLama2 BF16 Inference
TODO: @Liang use llama2 for below
For Llama, use the below command line to chat with it.
If you encounter a failure with the Llama fast tokenizer while using the latest transformers, add the option "--use_slow_tokenizer".
The `tokenizer_class` in `tokenizer_config.json` should be changed from `LLaMATokenizer` to `LlamaTokenizer`.
The `architectures` in `config.json` should be changed from `LLaMAForCausalLM` to `LlamaForCausalLM`.

```bash
python generate.py \
        --base_model_path "decapoda-research/llama-7b-hf" \
        --peft_model_path "./llama_peft_finetuned_model" \
        --use_slow_tokenizer \
        --use_kv_cache \
        --instructions "Transform the following sentence into one that shows contrast. The tree is rotten."
```

## LLama2 INT8 Inference
TODO: @chang add INT8 inference

Here are the explanations of each parameter:
`--temperature`: Controls the diversity of generated text. Lower values result in more deterministic outputs. The default value is 0.1.
`--top_p`: During text generation, only consider tokens with cumulative probability up to this value. This parameter helps to avoid extremely low probability tokens. The default value is 0.75.
`--top_k`: The number of highest probability vocabulary tokens to consider for each step of text generation. The default value is 40.
`--num_beams`: The number of beams to use for beam search decoding. This parameter helps to generate multiple possible completions. The default value is 1.
`--repetition_penalty`: This value controls the penalty for repeating tokens in the generated text. Higher values encourage the model to produce more diverse outputs. The default value is 1.1.
`--max_new_tokens`: The maximum number of tokens allowed in the generated output. This parameter helps to limit the length of the generated text. The default value is 128.


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
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host -v $(pwd):/intel-extension-for-transformers vault.habana.ai/gaudi-docker/1.11.0/ubuntu22.04/habanalabs/pytorch-installer-2.0.1:latest
apt-get update
apt-get install git-lfs
git-lfs install
cd /intel-extension-for-transformers/workflows/chatbot/inference/
pip install datasets
pip install optimum
pip install git+https://github.com/huggingface/optimum-habana.git
pip install peft
pip install einops
pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.11.0
```

## Run the inference

You can use the [generate.py](./generate.py) script for performing direct inference on Habana Gaudi instance. We have enabled BF16 to speed up the inference. Please use the following command for inference.

```bash
python generate.py --base_model_path "mosaicml/mpt-7b-chat" \
             --habana \
             --tokenizer_name "EleutherAI/gpt-neox-20b" \
             --use_hpu_graphs \
             --use_kv_cache \
             --instructions "Transform the following sentence into one that shows contrast. The tree is rotten."
```

And you can use `deepspeed` to speedup the inference. currently, TP is not supported for mpt

```bash
python ../utils/gaudi_spawn.py --use_deepspeed --world_size 8 generate.py \
        --base_model_path "mosaicml/mpt-7b-chat" \
        --habana \
        --tokenizer_name "EleutherAI/gpt-neox-20b" \
        --use_hpu_graphs \
        --use_kv_cache \
        --instructions "Transform the following sentence into one that shows contrast. The tree is rotten."
```

Habana supports HPU graph mode for inference speedup, which is available for bloom, gpt2, opt, gptj, gpt_neox, mpt, llama. You can use the parameter `use_hpu_graphs` to speed up the inference.

```bash
python generate.py --base_model_path "EleutherAI/gpt-j-6b" \
             --habana \
             --use_kv_cache \
             --use_hpu_graphs \
             --tokenizer_name "EleutherAI/gpt-j-6b" \
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
