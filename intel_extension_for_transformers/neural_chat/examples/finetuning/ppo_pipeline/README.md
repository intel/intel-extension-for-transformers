Reinforcement Learning from Human Feedback (RLHF)
============

Models such as ChatGPT, GPT-4, and Claude are powerful language models that have been fine-tuned using a method called Reinforcement Learning from Human Feedback (RLHF) to be better aligned with how we expect them to behave and would like to use them. we run RLHF through three steps.

** Supervised Fine-tuning (SFT)  
** Reward / preference modeling (RM)  
** Reinforcement Learning from Human Feedback (RLHF)  

## 1. Environment

```shell
pip install -r requirements.txt
pip install transformers==4.34.1
```
>**Note**: Suggest using transformers no higher than 4.34.1


## 2. Prepare reference dataset

We select 12k examples from [Orca](https://arxiv.org/abs/2306.02707) style dataset [Open-Orca/OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca), and regard its completions that are generated from GPT-4 or GPT-3.5 as chosen response. Simply and automatically, we use llama-2-13b-chat model to generate corresponding reject responses. For details of the dataset, you can refer [Intel/orca_dpo_pairs](https://huggingface.co/datasets/Intel/orca_dpo_pairs)

## 3. Supervised Fine-tuning (SFT)

you could refer https://github.com/intel/intel-extension-for-transformers/tree/main/intel_extension_for_transformers/neural_chat/examples/finetuning/instruction

### Training on CUDA

```
torchrun --nnodes 1  --nproc_per_node 8  finetune_clm.py \
        --model_name_or_path "meta-llama/Llama-2-7b-chat-hf" \
        --bf16 True \
        --dataset_name "Intel/orca_dpo_pairs" \
        --dataset_concatenation \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 2000 \
        --save_total_limit 1 \
        --learning_rate 1e-4  \
        --logging_steps 1 \
        --do_train \
        --num_train_epochs 3 \
        --overwrite_output_dir \
        --log_level info \
        --output_dir ./llama2_7b_se \
        --peft lora \
        --use_fast_tokenizer false \
        --use_auth_token True
```

### Training on Habana

```
python gaudi_spawn.py \
        --world_size 8 --use_mpi finetune_clm.py \
        --model_name_or_path "meta-llama/Llama-2-7b-chat-hf" \
        --bf16 True \
        --dataset_name "Intel/orca_dpo_pairs" \
        --dataset_concatenation \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 2000 \
        --save_total_limit 1 \
        --learning_rate 1e-4  \
        --logging_steps 1 \
        --do_train \
        --num_train_epochs 3 \
        --overwrite_output_dir \
        --log_level info \
        --output_dir ./llama2_7b_se \
        --peft lora \
        --use_fast_tokenizer false \
        --device "hpu" \
        --use_habana \
        --use_lazy_mode \
        --use_auth_token True
```

merge the adapter:

```
python3 merge_peft_adapter.py --adapter_model_name=../instruction/llama2_7b_se  --output_name=llama2_7b_se
```

## 4. Reward / preference modeling (RM) Fine-tuning

### Training on CUDA

```
torchrun --nnodes 1  --nproc_per_node 8  reward_modeling.py --model_name_or_path meta-llama/Llama-2-7b-hf --output_dir llama2_7b_rm --log_level info  --num_train_epochs 3  --logging_steps 1  --per_device_train_batch_size 4 --hf_access_token xxxxxx
```

### Training on Habana

Follow install guidance in [optimum-habana](https://github.com/huggingface/optimum-habana)

single card finetune
```
python3 reward_modeling.py --model_name_or_path meta-llama/Llama-2-7b-hf --output_dir llama2_7b_rm  --log_level info  --num_train_epochs 3 --use_habana --use_lazy_mode --hf_access_token xxxxxx  --logging_steps 1 --per_device_train_batch_size 4
```

multi card finetunes
```
python ../instruction/gaudi_spawn.py --world_size 8 --use_mpi reward_modeling.py --model_name_or_path meta-llama/Llama-2-7b-hf --output_dir llama2_7b_rm  --log_level info  --num_train_epochs 3 --use_habana --use_lazy_mode --hf_access_token xxxxxx --ddp_find_unused_parameters True --logging_steps 1 --per_device_train_batch_size 4
```

## 5. Reinforcement Fine-tuning

### Training on CUDA
```
accelerate launch --multi_gpu --num_machines 1  --num_processes 8 rl_training.py --log_with=wandb --model_name=llama2_7b_se --reward_model_name=llama2_7b_rm --adafactor=False --tokenizer_name=llama2_7b_se  --save_freq=100 --output_max_length=128 --batch_size=8 --gradient_accumulation_steps=8 --batched_gen=True --ppo_epochs=4 --seed=0 --learning_rate=1.4e-5 --early_stopping=True --output_dir=llama-se-rl-finetune-128-8-8-1.4e-5_adam --hf_access_token xxxxxx
```

### Training on Habana

Follow install guidance in [optimum-habana](https://github.com/huggingface/optimum-habana)

single card finetune

```
python3 rl_training.py  --model_name=llama2_7b_se --reward_model_name=llama2_7b_rm --adafactor=False --tokenizer_name=llama2_7b_se --save_freq=100 --output_max_length=128 --batch_size=8 --mini_batch_size=1 --gradient_accumulation_steps=8 --batched_gen=True --ppo_epochs=4 --seed=0 --learning_rate=1.4e-5 --early_stopping=True --output_dir=llama-se-rl-finetune-128-8-8-1.4e-5_adam --hf_access_token xxxxxx --use_habana
```

multi card finetunes
```
python3 ../instruction/gaudi_spawn.py --world_size 8 --use_mpi rl_training.py  --model_name=llama2_7b_se --reward_model_name=llama2_7b_rm --adafactor=False --tokenizer_name=llama2_7b_se --save_freq=100 --output_max_length=128 --batch_size=8 --mini_batch_size=1 --gradient_accumulation_steps=8 --batched_gen=True --ppo_epochs=4 --seed=0 --learning_rate=1.4e-5 --early_stopping=True --output_dir=llama-se-rl-finetune-128-8-8-1.4e-5_adam --hf_access_token xxxxxx --use_habana
```
