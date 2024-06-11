# GPT-J fine-tuning and inference


1. [Introduction](#introduction)
2. [Get Started](#get-started)

# Introduction

GPT-J 6B is an open-source large language model (LLM) with 6B parameters. Like GPT-3, it is an autoregressive, decoder-only transformer model designed to solve natural language processing (NLP) tasks by predicting how a piece of text will continue from a prompt. It is pre-trained on the Pile dataset, using the Mesh Transformer JAX library in JAX to handle the parallelization scheme with a vocab size of 50257 tokens.

This example demonstrates an end-end LLM fine-tuning workflow using [Glue MNLI](https://huggingface.co/datasets/glue/viewer/mnli/train) dataset and [LoRA](https://arxiv.org/abs/2106.09685) (Low-Rank Adaptation) technique to optimize the fine-tuning time.

This workflow runs on Intel Xeon CPU platform and has been verified on Intel 4th Gen Xeon CPU.

### Dataset
The MNLI dataset consists of pairs of sentences, a premise and a hypothesis. The task is to predict the relation between the premise and the hypothesis, which can be:
Entailment: the premise entails the hypothesis, 
Contradiction: hypothesis contradicts the premise and
Neutral: hypothesis and premise are unrelated. The prompt given to the model combines both premise and hypothesis and the fine-tuned model generates/predicts the relation as a next token.

### Fine-tuning and Inference
GPTJ-6B model is finetuned for NLI (Natural Language Inference) task using Glue MNLI Dataset. LoRA (PEFT) technique is applied which decomposes the LLM's weight matrices into smaller, lower-rank matrices called LoRA adapters. This significantly reduces the number of trainable parameters and thus the fine-tuning time without impacting the performance.
The fine-tuned model is then evaluated using Hugging face pipeline API using 'validation' dataset achieving SOTA accuracy.


# Get Started

### 1. Download the Workflow Repository

Clone the GPT-J repo for fine-tuning
```
git clone https://github.com/intel/intel-extension-for-transformers/tree/main
cd intel_extension_for_transformers/intel_extension_for_transformers/neural_chat/examples/finetuning/text_generation
```

### 2. Create environment and install software packages

Create miniconda environment and install the required packages
```
conda create -n gptj_ft_env python=3.10
conda activate gptj_ft_env
pip install -r requirements.txt
```

Install the following for best performance:
```
conda install mkl mkl-include -y
conda install gperftools jemalloc==5.2.1 -c conda-forge -y
```

### 3. Prepare dataset
We use the [Glue MNLI](https://huggingface.co/datasets/glue/viewer/mnli/train) dataset from hugging face.

## Fine-tuning

### Single-node fine-tuning
Set the following:
```
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0
```

Set libiomp
```
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
```

Tcmalloc is a recommended malloc implementation that emphasizes fragmentation avoidance and scalable concurrency support.
```
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
```

Run command or use singlenode_runscript.sh

We also supported Distributed Data Parallel finetuning on single node and multi nodes settings. 
Below command is to run single node fine-tuning.


```
python finetune_clm.py \
        --model_name_or_path "EleutherAI/gpt-j-6B" \
        --bf16 True \
        --dataset_name  "glue" \
        --dataset_config_name, "mnli" \
        --dataset_concatenation \
        --config_name ./config.json \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --do_train \
        --do_eval \
        --learning_rate 3.3113761e-4 \
        --num_train_epochs 3 \
        --logging_steps 100 \
        --save_total_limit 2 \
        --overwrite_output_dir \
        --log_level info \
        --save_strategy epoch \
        --output_dir ./gptj_peft_finetuned_model \
        --peft lora \
        --lora_alpha 54 \
        --lora_target_modules q_proj,v_proj,k_proj,out_proj \
        --use_fast_tokenizer false \
        --use_cpu \
        --task completion \
        --max_train_samples 5000 \
        --max_eval_samples  500  

```
