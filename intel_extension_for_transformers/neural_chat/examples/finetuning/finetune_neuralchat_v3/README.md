# How to train Intel/neural-chat-7b-v3-1 on Intel Gaudi2

[Intel/neural-chat-7b-v3-1](https://huggingface.co/Intel/neural-chat-7b-v3-1) ranks top1 on the [HuggingFaceH4/open_llm_leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) comparing with all the submitted 7B models (date: 11/17/2023). In this tutorial, we would like to share the details for the training process.

Similar to most finetuning work, we mainly divide the training to two stages.
- First stage: Use supervised fine-tuning (SFT) to improve the performance of the base model like [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1).
- Second stage: apply Direct Preference Optimization (DPO) to align the SFT model with preference dataset.


## Prepare Environment

In order to streamline the process, users can construct a Docker image employing a Dockerfile, initiate the Docker container, and then proceed to execute inference or finetuning operations.

**IMPORTANT:** Please note Intel Gaudi2 processors(HPU) requires docker environment for running. User needs to manually execute below steps to build docker image and run docker container for inference on Intel Gaudi2. The Jupyter notebook server should be started in the docker container and then run this Jupyter notebook.

To run finetuning on Intel Gaudi2, please execute below steps

```bash
git clone https://github.com/intel/intel-extension-for-transformers.git
cd intel-extension-for-transformers

docker build --no-cache ./ --target hpu --build-arg REPO=https://github.com/intel/intel-extension-for-transformers.git --build-arg ITREX_VER=main -f ./intel_extension_for_transformers/neural_chat/docker/Dockerfile -t chatbot_finetuning:latest

docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host chatbot_finetuning:latest

# after entering docker container
cd examples/finetuning/finetune_neuralchat_v3

```


## Supervised Fine-Tuning (SFT)

We select the latest pretrained [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) and the open source dataset [Open-Orca/SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca) to conduct the experiment.

The below script use deepspeed zero2 to launch the training with 8 cards Gaudi2. In the `finetune_neuralchat_v3.py`, the default `use_habana=True, use_lazy_mode=True, device="hpu"` for Gaudi2. And if you want to run it on Nvidia GPU, you can set them `use_habana=False, use_lazy_mode=False, device="auto"`.

```python
deepspeed --include localhost:0,1,2,3,4,5,6,7 \
    --master_port 29501 \
    finetune_neuralchat_v3.py
```


#### Merge the lora weights

```python
python apply_lora.py \
    --base-model-path mistralai/Mistral-7B-v0.1 \
    --lora-model-path finetuned_model/ \
    --output-path finetuned_model_lora
```

## Direct Preference Optimization (DPO)

For Better aligning human preferences, we apply [Direct Preference Optimization](https://arxiv.org/pdf/2305.18290.pdf) (DPO) algorithm, which is stable and computationally lightweight. The algorithm derives the probability of human preference data for the optimal policy to replace the reward model that reinforcement learning from human feedback (RLHF) needs and formulates a maximum likelihood objective for a parameterized policy.

The preference dataset contains 12k examples selected from [Orca](https://arxiv.org/abs/2306.02707) style dataset [Open-Orca/OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca) and its completions that are generated from GPT-4 or GPT-3.5 are regarded as chosen response. In term of reject responses, we use [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) model to generate corresponding completions automatically because maybe better rejected responses are also better for alignment. 

For details of the dataset and DPO training code, you can refer [Intel/orca_dpo_pairs](https://huggingface.co/datasets/Intel/orca_dpo_pairs) and [DPO example](https://github.com/intel/intel-extension-for-transformers/tree/main/intel_extension_for_transformers/neural_chat/examples/finetuning/dpo_pipeline).


```python
python ../dpo_pipeline/dpo_clm.py \
        --model_name_or_path "./finetuned_model_lora" \
        --output_dir "./finetuned_model_lora_plus_dpo" \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 8 \
        --learning_rate 5e-4 \
        --max_steps 1000 \
        --save_steps 10 \
        --logging_steps 10 \
        --lora_alpha 16 \
        --lora_rank 16 \
        --lora_dropout 0.05 \
        --dataset_name Intel/orca_dpo_pairs \
        --bf16 \
        --max_length 1536 \
        --max_prompt_length 1024 \
        --lr_scheduler_type "cosine" \
        --warmup_steps 100 \
        --use_habana \
        --use_lazy_mode \
        --pad_max true \
        --gradient_checkpointing true \
        --lora_all_linear false \
        --lora_target_modules 'k_proj' 'q_proj' 'v_proj'
```
