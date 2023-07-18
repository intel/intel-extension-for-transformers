Intel Chatbot Finetuning Dockerfile installer for Ubuntu22.04

# Prerequisite​

## 1. Prepare the Model

### LLaMA
To acquire the checkpoints and tokenizer, the user has two options: completing the [Google form](https://forms.gle/jk851eBVbX1m5TAv5) or attempting [the released model on Huggingface](https://huggingface.co/decapoda-research/llama-7b-hf). 

It should be noticed that the early version of LLama model's name in Transformers has resulted in many loading issues, please refer to this [revision history](https://github.com/huggingface/transformers/pull/21955). Therefore, Transformers has reorganized the code and rename LLaMA model as `Llama` in the model file. But the release model on Huggingface did not make modifications in react to this change. To avoid unexpexted confliction issues, we advise the user to modify the local `config.json` and `tokenizer_config.json` files according to the following recommendations:
1. The `tokenizer_class` in `tokenizer_config.json` should be changed from `LLaMATokenizer` to `LlamaTokenizer`;
2. The `architectures` in `config.json` should be changed from `LLaMAForCausalLM` to `LlamaForCausalLM`.

### FLAN-T5
The user can obtain the [release model](https://huggingface.co/google/flan-t5-xl) from Huggingface.

## 2. Prepare Dataset
The instruction-following dataset is needed for the finetuning. We select two kinds of Datasets to conduct the finetuning process: general domain dataset and domain specific dataset.

1. General domain dataset: We use the [Alpaca dataset](https://github.com/tatsu-lab/stanford_alpaca) from Stanford University as the general domain dataset to fine-tune the model. This dataset is provided in the form of a JSON file, [alpaca_data.json](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json). In Alpaca, researchers have manually crafted 175 seed tasks to guide `text-davinci-003` in generating 52K instruction data for diverse tasks.

2. Domain-specific dataset: Inspired by Alpaca, we constructed a domain-specific dataset focusing on Business and Intel-related issues. We made minor modifications to the [prompt template](https://github.com/tatsu-lab/stanford_alpaca/blob/main/prompt.txt) to proactively guide Alpaca in generating more Intel and Business related instruction data. The generated data could be find in `intel_domain.json`.

## 3. Prepare Dockerfile
Assuming you have downloaded the model and dataset to your workspace /path/to/workspace/
Please clone a ITREX repo to this path.
```bash
git clone https://github.com/intel-innersource/frameworks.ai.nlp-toolkit.intel-nlp-toolkit.git
```
You can modify the model path at line 70 if you are going to run other models:
```bash
vim /path/to/workspace/frameworks.ai.nlp-toolkit.intel-nlp-toolkit/workflows/chatbot/fine_tuning/docker/Dockerfile
```
```
COPY flan-t5-xl /flan/
```

## 4. Build Docker Image
```bash
docker build --build-arg UBUNTU_VER=22.04 --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f  /path/to/workspace/frameworks.ai.nlp-toolkit.intel-nlp-toolkit/workflows/chatbot/fine_tuning/docker/Dockerfile -t chatbot_finetune .
```
## 5. Create Docker Container
docker run -tid --disable-content-trust --privileged --name="chatbot" --hostname="chatbot-container" --network=host -e https_proxy -e http_proxy -e HTTPS_PROXY -e HTTP_PROXY -e no_proxy -e NO_PROXY -v /dev/shm:/dev/shm "chatbot_finetune"

# Finetune

We employ the [LoRA approach](https://arxiv.org/pdf/2106.09685.pdf) to finetune the LLM efficiently, currently, FLAN-T5 and LLaMA are supported for finetuning.

## 1. Single Node Fine-tuning

For FLAN-T5, use the below command line for finetuning on the Alpaca dataset.

```bash
docker exec "chatbot" bash -c "
timeout 10800 python ./fine_tuning/instruction_tuning_pipeline/finetune_seq2seq.py \
        --model_name_or_path "/flan" \
        --train_file "/dataset/alpaca_data.json" \
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
        --peft lora"
```

For LLaMA, use the below command line for finetuning on the Alpaca dataset.

```bash
docker exec "chatbot" bash -c "
python ./fine_tuning/instruction_tuning_pipeline/finetune_clm.py \
        --model_name_or_path "/llama_7b" \
        --train_file "/dataset/alpaca_data.json" \
        --dataset_concatenation \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --do_train \
        --learning_rate 2e-5 \
        --num_train_epochs 3 \
        --logging_steps 100 \
        --save_total_limit 2 \
        --overwrite_output_dir \
        --log_level info \
        --save_strategy epoch \
        --output_dir ./llama_peft_finetuned_model \
        --peft lora \
        --use_fast_tokenizer false"
```

Where the `--dataset_concatenation` argument is a way to vastly accelerate the fine-tuning process through training samples concatenation. With several tokenized sentences concatenated into a longer and concentrated sentence as the training sample instead of having several training samples with different lengths, this way is more efficient due to the parallelism characteristic provided by the more concentrated training samples.

For finetuning on SPR, add `--bf16` argument will speedup the finetuning process without the loss of model's performance.

## 2. Multi-node Fine-tuning

We also supported Distributed Data Parallel finetuning on single node and multi-node settings. To use Distributed Data Parallel to speedup training, the bash command needs a small adjustment. Note for multi-node setting, steps of [Prerequisite​](#Prerequisite​) need to be done on each node.
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
> Also please note that to use CPU for training in each node with multi-node settings, argument `--no_cuda` is mandatory, and `--xpu_backend ccl` is required if to use ccl as the distributed backend. In multi-node setting, following command needs to be launched in each node, and all the commands should be the same except for *`<NODE_RANK>`*, which should be integer from 0 to *`<NUM_NODES>`*`-1` assigned to each node.

``` bash
docker exec "chatbot" bash -c "
torchrun --master_addr=<MASTER_ADDRESS> --nproc_per_node=<NUM_PROCESSES_PER_NODE> --nnodes=<NUM_NODES> --node_rank=<NODE_RANK> \
        ./fine_tuning/instruction_tuning_pipeline/finetune_seq2seq.py \
        --model_name_or_path "google/flan-t5-xl" \
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
        --peft lora"
```

For LLaMA, use the below command line for finetuning on the Alpaca dataset through Distributed Data Parallel training.

```bash
docker exec "chatbot" bash -c "
torchrun --master_addr=<MASTER_ADDRESS> --nproc_per_node=<NUM_PROCESSES_PER_NODE> --nnodes=<NUM_NODES> --node_rank=<NODE_RANK> \
        ./fine_tuning/instruction_tuning_pipeline/finetune_clm.py \
        --model_name_or_path "/llama_7b" \
        --train_file "/dataset/alpaca_data.json" \
        --dataset_concatenation \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --do_train \
        --learning_rate 2e-5 \
        --num_train_epochs 3 \
        --logging_steps 100 \
        --save_total_limit 2 \
        --overwrite_output_dir \
        --log_level info \
        --save_strategy epoch \
        --output_dir ./llama_peft_finetuned_model \
        --peft lora \
        --use_fast_tokenizer false"
```