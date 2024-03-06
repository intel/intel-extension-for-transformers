NeuralChat Fine-tuning
============

This example demonstrates how to finetune the pretrained large language model (LLM) with the instruction-following dataset for creating the NeuralChat, a chatbot that can conduct the textual conversation. Giving NeuralChat the textual instruction, it will respond with the textual response. This example have been validated on the 4th Gen Intel® Xeon® Processors, Sapphire Rapids.

## Validated Model List
|Pretrained model| Text Generation (Instruction) | Text Generation (ChatBot) | Summarization | Code Generation | 
|------------------------------------|---|---|--- | --- |
|LLaMA series| ✅| ✅|✅| ✅
|LLaMA2 series| ✅| ✅|✅| ✅
|MPT series| ✅| ✅|✅| ✅
|FLAN-T5 series| ✅ | **WIP**| **WIP** | **WIP**|

# Prerequisite​

## 1. Environment​
Recommend python 3.9 or higher version.
```shell
pip install -r requirements.txt
# To use ccl as the distributed backend in distributed training on CPU requires to install below requirement.
python -m pip install oneccl_bind_pt==2.2.0 -f https://developer.intel.com/ipex-whl-stable-cpu
```

## 2. Prepare the Model

### LLaMA
To acquire the checkpoints and tokenizer, the user has two options: completing the [Google form](https://forms.gle/jk851eBVbX1m5TAv5) or attempting [the released model on Huggingface](https://huggingface.co/decapoda-research/llama-7b-hf). 

It should be noticed that the early version of LLama model's name in Transformers has resulted in many loading issues, please refer to this [revision history](https://github.com/huggingface/transformers/pull/21955). Therefore, Transformers has reorganized the code and rename LLaMA model as `Llama` in the model file. But the release model on Huggingface did not make modifications in react to this change. To avoid unexpected confliction issues, we advise the user to modify the local `config.json` and `tokenizer_config.json` files according to the following recommendations:
1. The `tokenizer_class` in `tokenizer_config.json` should be changed from `LLaMATokenizer` to `LlamaTokenizer`;
2. The `architectures` in `config.json` should be changed from `LLaMAForCausalLM` to `LlamaForCausalLM`.

### FLAN-T5
The user can obtain the [release model](https://huggingface.co/google/flan-t5-xl) from Huggingface.

## 3. Prepare Dataset
We select 4 kind of datasets to conduct the finetuning process for different tasks.

1. Text Generation (General domain instruction): We use the [Alpaca dataset](https://github.com/tatsu-lab/stanford_alpaca) from Stanford University as the general domain dataset to fine-tune the model. This dataset is provided in the form of a JSON file, [alpaca_data.json](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json). In Alpaca, researchers have manually crafted 175 seed tasks to guide `text-davinci-003` in generating 52K instruction data for diverse tasks.

    - **Completion data format:** the data fields are `instruction, input, output` ([Alpaca dataset](https://github.com/tatsu-lab/stanford_alpaca)), one of examples is:
    ```python
    # for examples with a non-empty input field
    {
        "instruction": "Explain why the following fraction is equivalent to 1/4",
        "input": "4/16",
        "output": "The fraction 4/16 is equivalent to 1/4 because both numerators and denominators are divisible by 4. Dividing both the top and bottom numbers by 4 yields the fraction 1/4."
    }
    # for examples with a empty input field 
    {
        "instruction": "Give three tips for staying healthy.",
        "input": "",
        "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."
    }
    ```

    - **Completion template:**
        - for examples with a non-empty input field:
        ```python
        Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        {instruction}
    
        ### Input:
        {input}

        ### Response:

        ```
        - for examples with a empty input field:
        ```python
        Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        {instruction}


        ### Response:

        ```

2. Text Generation (Domain-specific instruction): Inspired by Alpaca, we constructed a domain-specific dataset focusing on Business and Intel-related issues. We made minor modifications to the [prompt template](https://github.com/tatsu-lab/stanford_alpaca/blob/main/prompt.txt) to proactively guide Alpaca in generating more Intel and Business related instruction data. The generated data could be find in `intel_domain.json`. The `data format` and `template` is same with `Instruction data format` and `Instruction template`.

3. Text Generation (ChatBot): To finetune a chatbot, we use the chat-style dataset [HuggingFaceH4/oasst1_en](https://huggingface.co/datasets/HuggingFaceH4/oasst1_en).

    - **Chat data format:** the data field is `messages` (keep same with [HuggingFaceH4/oasst1_en](https://huggingface.co/datasets/HuggingFaceH4/oasst1_en)), one of examples is:
    ```python
    {
        "messages":[
            {
                "content":"Can you give me a list of names for the title of a project I am working on? The project is a video game I am working on for school about a zombie apocalypse that happens in tribal times. And can you include these words in some way: \"Death, fire, tribal, ancient\"",
                "role":"user"
            },
            {
                "content":"Here are some potential names:\n\nThe ancient fire and the Tribe of death, \nthe Tribe of Fire and the Tribe of ghosts,\nThe curse of the fire of death,\nThe curse of the flaming Angels,\nThe black flame,",
                "role":"assistant"
            }
        ]
    }
    ```

    - **Chat template:**
    ```python
    <|im_start|>system
    - You are a helpful assistant chatbot trained by Intel.
    - You answer questions.
    - You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
    - You are more than just an information source, you are also able to write poetry, short stories, and make jokes.<|im_end|>\n
    {single/multi turn conversation}
    ```

4. Summarization: An English-language dataset [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) containing just over 300k unique news articles as written by journalists at CNN and the Daily Mail, is used for this task.

    - **Summarization data format:** the main data fields are `article` and `highlights` (keep same with [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail)), one of examples is:
    ```python
    {
        "article":"LONDON, England (Reuters) -- Harry Potter star Daniel Radcliffe gains access to a reported £20 million ($41.1 million) fortune as he turns 18 on Monday, but he insists the money won't cast a spell on him. Daniel Radcliffe as Harry Potter in Harry Potter and the Order of the Phoenix To the disappointment of gossip columnists around the world, the young actor says he has no plans to fritter his cash away on fast cars, drink and celebrity parties......",
        "highlights":"Harry Potter star Daniel Radcliffe gets £20M fortune as he turns 18 Monday. Young actor says he has no plans to fritter his cash away. Radcliffe's earnings from first five Potter films have been held in trust fund."
    }
    ```

    - **Summarization template:**
    ```python
    {article}\nSummarize the highlights of this article.\n{highlights}
    ```


5. Code Generation: To enhance code performance of LLMs (Large Language Models), we use the [theblackcat102/evol-codealpaca-v1](https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1). The `data format` and `template` is same with `Instruction data format` and `Instruction template`.


**note:** for more details about preprocessing and customizing dataset, please refer `instruction_tuning_pipeline/data_utils.py`


# Finetune

We employ the [LoRA approach](https://arxiv.org/pdf/2106.09685.pdf) to finetune the LLM efficiently, currently, FLAN-T5 and LLaMA are supported for finetuning.

## 1. Single Node Fine-tuning in Xeon SPR

**For FLAN-T5**, use the below command line for finetuning on the Alpaca dataset.

```bash
python finetune_seq2seq.py \
        --model_name_or_path "google/flan-t5-xl" \
        --bf16 True \
        --train_file "stanford_alpaca/alpaca_data.json" \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 1 \
        --do_train \
        --learning_rate 1.0e-5 \
        --warmup_ratio 0.03 \
        --weight_decay 0.0 \
        --num_train_epochs 5 \
        --logging_steps 10 \
        --save_steps 2000 \
        --save_total_limit 2 \
        --overwrite_output_dir \
        --output_dir ./flan-t5-xl_peft_finetuned_model \
        --peft lora
```

#### For LLaMA 

- use the below command line for finetuning on the Alpaca dataset.

```bash
python finetune_clm.py \
        --model_name_or_path "decapoda-research/llama-7b-hf" \
        --bf16 True \
        --train_file "/path/to/alpaca_data.json" \
        --dataset_concatenation \
        --task completion \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --do_train \
        --learning_rate 1e-4 \
        --num_train_epochs 3 \
        --logging_steps 100 \
        --save_total_limit 2 \
        --overwrite_output_dir \
        --log_level info \
        --save_strategy epoch \
        --output_dir ./llama_peft_finetuned_model \
        --peft lora \
        --use_fast_tokenizer false \
        --no_cuda \
```

**note:** set `--do_lm_eval` to evaluate model with `truthfulqa_mc` metric, and you can set `--lm_eval_tasks` to evaluate more tasks supported in [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)


- use the below command line for finetuning chatbot on the [HuggingFaceH4/oasst1_en](https://huggingface.co/datasets/HuggingFaceH4/oasst1_en).

```bash
python finetune_clm.py \
        --model_name_or_path "decapoda-research/llama-7b-hf" \
        --bf16 True \
        --dataset_name "HuggingFaceH4/oasst1_en" \
        --task chat \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --do_train \
        --learning_rate 1e-4 \
        --num_train_epochs 3 \
        --logging_steps 100 \
        --save_total_limit 2 \
        --overwrite_output_dir \
        --log_level info \
        --save_strategy epoch \
        --output_dir ./llama_chatbot_peft_finetuned_model \
        --peft lora \
        --use_fast_tokenizer false \
        --no_cuda \

# the script also support other models, like mpt.
```

- use the below command line for summarization scenario on the [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail).

```bash
python finetune_clm.py \
        --model_name_or_path "decapoda-research/llama-7b-hf" \
        --bf16 True \
        --dataset_name "cnn_dailymail" \
        --dataset_config_name "3.0.0" \
        --task summarization \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --do_train \
        --learning_rate 1e-4 \
        --num_train_epochs 3 \
        --logging_steps 100 \
        --save_total_limit 2 \
        --overwrite_output_dir \
        --log_level info \
        --save_strategy epoch \
        --output_dir ./llama_peft_finetuned_model \
        --peft lora \
        --use_fast_tokenizer false \
        --no_cuda

# the script also support other models, like mpt.
```

**note:** Use `rouge` metric to evaluate model on summarization task.

- use the below command line for code tuning with `meta-llama/Llama-2-7b-hf` on [theblackcat102/evol-codealpaca-v1](https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1).

```bash
python finetune_clm.py \
        --model_name_or_path "meta-llama/Llama-2-7b-hf" \
        --bf16 True \
        --dataset_name "theblackcat102/evol-codealpaca-v1" \
        --task completion \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --do_train \
        --learning_rate 1e-4 \
        --num_train_epochs 3 \
        --logging_steps 100 \
        --save_total_limit 2 \
        --overwrite_output_dir \
        --log_level info \
        --save_strategy epoch \
        --output_dir ./llama_peft_finetuned_model \
        --peft lora \
        --use_fast_tokenizer false \
        --no_cuda

# the script also support other models, like mpt.
```



**For [MPT](https://huggingface.co/mosaicml/mpt-7b)**, use the below command line for finetuning on the Alpaca dataset. Only LORA supports MPT in PEFT perspective.it uses gpt-neox-20b tokenizer, so you need to define it in command line explicitly.

```bash
python finetune_clm.py \
        --model_name_or_path "mosaicml/mpt-7b" \
        --bf16 True \
        --train_file "/path/to/alpaca_data.json" \
        --task completion \
        --dataset_concatenation \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --do_train \
        --learning_rate 1e-4 \
        --num_train_epochs 3 \
        --logging_steps 100 \
        --save_total_limit 2 \
        --overwrite_output_dir \
        --log_level info \
        --save_strategy epoch \
        --output_dir ./mpt_peft_finetuned_model \
        --peft lora \
        --tokenizer_name "EleutherAI/gpt-neox-20b" \
        --no_cuda \
```

Where the `--dataset_concatenation` argument is a way to vastly accelerate the fine-tuning process through training samples concatenation. With several tokenized sentences concatenated into a longer and concentrated sentence as the training sample instead of having several training samples with different lengths, this way is more efficient due to the parallelism characteristic provided by the more concentrated training samples.

For finetuning on SPR, add `--bf16` argument will speedup the finetuning process without the loss of model's performance.
You could also indicate `--peft` to switch peft method in P-tuning, Prefix tuning, Prompt tuning, LLama Adapter, LoRA,
see https://github.com/huggingface/peft. Note for FLAN-T5/MPT, only LoRA is supported.

Add option **"--use_fast_tokenizer False"** when using latest transformers if you met failure in llama fast tokenizer for llama, The `tokenizer_class` in `tokenizer_config.json` should be changed from `LLaMATokenizer` to `LlamaTokenizer`

## 2. Multi-node Fine-tuning in Xeon SPR

We also supported Distributed Data Parallel finetuning on single node and multi-node settings. To use Distributed Data Parallel to speedup training, the bash command needs a small adjustment.
<br>
For example, to finetune FLAN-T5 through Distributed Data Parallel training, bash command will look like the following, where
<br>
*`<MASTER_ADDRESS>`* is the address of the master node, it won't be necessary for single node case,
<br>
*`<NUM_PROCESSES_PER_NODE>`* is the desired processes to use in current node, for node with GPU, usually set to number of GPUs in this node, for node without GPU and use CPU for training, it's recommended set to 1,
<br>
*`<NUM_NODES>`* is the number of nodes to use,
<br>
*`<NODE_RANK>`* is the rank of the current node, rank starts from 0 to *`<NUM_NODES>`*`-1`.
<br>
> Also please note that to use CPU for training in each node with multi-node settings, argument `--no_cuda` is mandatory, and `--ddp_backend ccl` is required if to use ccl as the distributed backend. In multi-node setting, following command needs to be launched in each node, and all the commands should be the same except for *`<NODE_RANK>`*, which should be integer from 0 to *`<NUM_NODES>`*`-1` assigned to each node.

``` bash
mpirun -f nodefile -n 16 -ppn 4 -genv OMP_NUM_THREADS=56 python3 finetune_seq2seq.py \
    --model_name_or_path "google/flan-t5-xl" \
    --bf16 True \
    --train_file "stanford_alpaca/alpaca_data.json" \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --do_train \
    --learning_rate 1.0e-5 \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 5 \
    --logging_steps 10 \
    --save_steps 2000 \
    --save_total_limit 2 \
    --overwrite_output_dir \
    --output_dir ./flan-t5-xl_peft_finetuned_model \
    --peft lora \
    --no_cuda \
    --ddp_backend ccl \
```
If you have enabled passwordless SSH in cpu clusters, you could also use mpirun in master node to start the DDP finetune. Take llama alpaca finetune for example. follow the [hugginface guide](https://huggingface.co/docs/transformers/perf_train_cpu_many) to install Intel® oneCCL Bindings for PyTorch, IPEX

oneccl_bindings_for_pytorch is installed along with the MPI tool set. Need to source the environment before using it.

for Intel® oneCCL >= 1.12.0
``` bash
oneccl_bindings_for_pytorch_path=$(python -c "from oneccl_bindings_for_pytorch import cwd; print(cwd)")
source $oneccl_bindings_for_pytorch_path/env/setvars.sh
```

for Intel® oneCCL whose version < 1.12.0
``` bash
torch_ccl_path=$(python -c "import torch; import torch_ccl; import os;  print(os.path.abspath(os.path.dirname(torch_ccl.__file__)))")
source $torch_ccl_path/env/setvars.sh
```

The following command enables training with a total of 16 processes on 4 Xeons (node0/1/2/3, 2 sockets each node. taking node0 as the master node), ppn (processes per node) is set to 4, with two processes running per one socket. The variables OMP_NUM_THREADS/CCL_WORKER_COUNT can be tuned for optimal performance.

In node0, you need to create a configuration file which contains the IP addresses of each node (for example hostfile) and pass that configuration file path as an argument.
``` bash
 cat hostfile
 xxx.xxx.xxx.xxx #node0 ip
 xxx.xxx.xxx.xxx #node1 ip
 xxx.xxx.xxx.xxx #node2 ip
 xxx.xxx.xxx.xxx #node3 ip
```
Now, run the following command in node0 and **4DDP** will be enabled in node0 and node1 with BF16 auto mixed precision:
``` bash
export CCL_WORKER_COUNT=1
export MASTER_ADDR=xxx.xxx.xxx.xxx #node0 ip
## for DDP ptun for LLama
mpirun -f nodefile -n 16 -ppn 4 -genv OMP_NUM_THREADS=56 python3 finetune_clm.py \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --train_file ./alpaca_data.json \
    --task completion \
    --bf16 True \
    --output_dir ./llama_peft_finetuned_model \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --logging_steps 1 \
    --peft ptun \
    --group_by_length True \
    --dataset_concatenation \
    --use_fast_tokenizer false \
    --do_train \
    --no_cuda \
    --ddp_backend ccl \

## for DDP LORA for MPT
mpirun -f nodefile -n 16 -ppn 4 -genv OMP_NUM_THREADS=56 python3 finetune_clm.py \
    --model_name_or_path mosaicml/mpt-7b \
    --train_file ./alpaca_data.json \
    --task completion \
    --bf16 True \
    --output_dir ./mpt_peft_finetuned_model \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 1e-4  \
    --logging_steps 1 \
    --peft lora \
    --group_by_length True \
    --dataset_concatenation \
    --do_train \
    --tokenizer_name "EleutherAI/gpt-neox-20b" \
    --no_cuda \
    --ddp_backend ccl \
```
you could also indicate `--peft` to switch peft method in P-tuning, Prefix tuning, Prompt tuning, LLama Adapter, LORA,
see https://github.com/huggingface/peft

## 1. Single Card Fine-tuning in Habana DL1

Follow install guidance in [optimum-habana](https://github.com/huggingface/optimum-habana)

For LLaMA, use the below command line for finetuning on the Alpaca dataset.

```bash
python finetune_clm.py \
        --model_name_or_path "decapoda-research/llama-7b-hf" \
        --bf16 True \
        --train_file "/path/to/alpaca_data.json" \
        --task completion \
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
        --output_dir ./llama_peft_finetuned_model \
        --peft lora \
        --use_fast_tokenizer false \
        --device hpu \
        --use_habana \
        --use_lazy_mode \
```

For [MPT](https://huggingface.co/mosaicml/mpt-7b), use the below command line for finetuning on the Alpaca dataset. Only LORA supports MPT in PEFT perspective.it uses gpt-neox-20b tokenizer, so you need to define it in command line explicitly.

```bash
python finetune_clm.py \
        --model_name_or_path "mosaicml/mpt-7b" \
        --bf16 True \
        --train_file "/path/to/alpaca_data.json" \
         --task completion \
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
        --output_dir ./mpt_peft_finetuned_model \
        --peft lora \
        --tokenizer_name "EleutherAI/gpt-neox-20b" \
        --device hpu \
        --use_habana \
        --use_lazy_mode \
```
Where the `--dataset_concatenation` argument is a way to vastly accelerate the fine-tuning process through training samples concatenation. With several tokenized sentences concatenated into a longer and concentrated sentence as the training sample instead of having several training samples with different lengths, this way is more efficient due to the parallelism characteristic provided by the more concentrated training samples.

For finetuning on SPR, add `--bf16` argument will speedup the finetuning process without the loss of model's performance.
You could also indicate `--peft` to switch peft method in P-tuning, Prefix tuning, Prompt tuning, LLama Adapter, LoRA,
see https://github.com/huggingface/peft. Note for MPT, only LoRA is supported.

Add option **"--use_fast_tokenizer False"** when using latest transformers if you met failure in llama fast tokenizer for llama, The `tokenizer_class` in `tokenizer_config.json` should be changed from `LLaMATokenizer` to `LlamaTokenizer`


## 2. Multi Card Fine-tuning in Habana DL1

Follow install guidance in [optimum-habana](https://github.com/huggingface/optimum-habana)

For LLaMA, use the below command line for finetuning on the Alpaca dataset.

```bash
python ../../utils/gaudi_spawn.py \
        --world_size 8 --use_mpi finetune_clm.py \
        --model_name_or_path "decapoda-research/llama-7b-hf" \
        --bf16 True \
        --train_file "/path/to/alpaca_data.json" \
        --task completion \
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
        --output_dir ./llama_peft_finetuned_model \
        --peft lora \
        --use_fast_tokenizer false \
        --device hpu \
        --use_habana \
        --use_lazy_mode \
        --distribution_strategy fast_ddp \
```

For [MPT](https://huggingface.co/mosaicml/mpt-7b), use the below command line for finetuning on the Alpaca dataset. Only LORA supports MPT in PEFT perspective.it uses gpt-neox-20b tokenizer, so you need to define it in command line explicitly.

```bash
python ../../utils/gaudi_spawn.py \
        --world_size 8 --use_mpi finetune_clm.py \
        --model_name_or_path "mosaicml/mpt-7b" \
        --bf16 True \
        --train_file "/path/to/alpaca_data.json" \
        --task completion \
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
        --output_dir ./mpt_peft_finetuned_model \
        --peft lora \
        --tokenizer_name "EleutherAI/gpt-neox-20b" \
        --device hpu \
        --use_habana \
        --use_lazy_mode \
        --distribution_strategy fast_ddp \
```

Where the `--dataset_concatenation` argument is a way to vastly accelerate the fine-tuning process through training samples concatenation. With several tokenized sentences concatenated into a longer and concentrated sentence as the training sample instead of having several training samples with different lengths, this way is more efficient due to the parallelism characteristic provided by the more concentrated training samples.

For finetuning on SPR, add `--bf16` argument will speedup the finetuning process without the loss of model's performance.
You could also indicate `--peft` to switch peft method in P-tuning, Prefix tuning, Prompt tuning, LLama Adapter, LoRA,
see https://github.com/huggingface/peft. Note for MPT, only LoRA is supported.

Add option **"--use_fast_tokenizer False"** when using latest transformers if you met failure in llama fast tokenizer for llama, The `tokenizer_class` in `tokenizer_config.json` should be changed from `LLaMATokenizer` to `LlamaTokenizer`

# Evaluation

- For task=completion/chat, set `--do_lm_eval` to evaluate model with `truthfulqa_mc` metric, and you can set `--lm_eval_tasks` to evaluate more tasks supported in [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- For task=summarization, we use `rouge` metric.
- For custom evaluation function, you can refer to `instruction_tuning_pipeline/eval_utils.py`, and call it at end of the training
 
