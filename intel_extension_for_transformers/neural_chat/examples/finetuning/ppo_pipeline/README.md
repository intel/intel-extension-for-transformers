Reinforcement Learning from Human Feedback (RLHF)
============

Models such as ChatGPT, GPT-4, and Claude are powerful language models that have been fine-tuned using a method called Reinforcement Learning from Human Feedback (RLHF) to be better aligned with how we expect them to behave and would like to use them. we run RLHF through three steps.

** Supervised Fine-tuning (SFT)  
** Reward / preference modeling (RM)  
** Reinforcement Learning from Human Feedback (RLHF)  

## 1. Environment

```shell
pip install -r requirements.txt
```

## 2. Prepare reference dataset

We select 12k examples from [Orca](https://arxiv.org/abs/2306.02707) style dataset [Open-Orca/OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca), and regard its completions that are generated from GPT-4 or GPT-3.5 as chosen response. Simply and automatically, we use llama-2-13b-chat model to generate corresponding reject responses. For details of the dataset, you can refer [Intel/orca_dpo_pairs](https://huggingface.co/datasets/Intel/orca_dpo_pairs)

## 3. Supervised Fine-tuning (SFT)

you could refer https://github.com/intel/intel-extension-for-transformers/tree/main/intel_extension_for_transformers/neural_chat/examples/finetuning/instruction


## 4. Reward / preference modeling (RM) Fine-tuning

### Training on CUDA

```
torchrun --nnodes 1  --nproc_per_node 8  reward_modeling.py --model_name_or_path meta-llama/Llama-2-7b-hf --output_dir <output> --log_level info  --num_train_epochs 1  --per_device_train_batch_size 1 --hf_access_token xxxxxx
```

### Training on Habana

Follow install guidance in [optimum-habana](https://github.com/huggingface/optimum-habana)

single card finetune
```
python3 reward_modeling.py --model_name_or_path meta-llama/Llama-2-7b-hf --output_dir <output>  --log_level info  --num_train_epochs 1 --use_habana --use_lazy_mode --hf_access_token xxxxxx
```

multi card finetunes
```
python ../instruction/gaudi_spawn.py --world_size 8 --use_mpi reward_modeling.py --model_name_or_path meta-llama/Llama-2-7b-hf --output_dir <output>  --log_level info  --num_train_epochs 1 --use_habana --use_lazy_mode --hf_access_token xxxxxx --ddp_find_unused_parameters True
```
