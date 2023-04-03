NeuralChat
============

This example demonstrates how to finetune the pretrained large language model (LLM) with the instruction-following dataset for creating the NeuralChat, a chatbot that can conduct the textual conversation. Giving NeuralChat the textual instruction, it will respond with the textual response. This example have been validated on the 4th Gen Intel® Xeon® Processors, Sapphire Rapids.

# Prerequisite​

## 1. Environment​
Recommend python 3.7 or higher version.
```shell
pip install -r requirements.txt
```

## 2. Prepare the Model

### LLaMA
To acquire the checkpoints and tokenizer, the user has two options: completing the [Google form](https://forms.gle/jk851eBVbX1m5TAv5) or attempting [the released model on Huggingface](https://huggingface.co/decapoda-research/llama-7b-hf). 

It should be noticed that the early version of LLama model's name in Transformers has resulted in many loading issues, please refer to this [revision history](https://github.com/huggingface/transformers/pull/21955). Therefore, Transformers has reorganized the code and rename LLaMA model as `Llama` in the model file. But the release model on Huggingface did not make modifications in react to this change. To avoid unexpexted confliction issues, we advise the user to modify the local `config.json` and `tokenizer_config.json` files according to the following recommendations:
1. The `tokenizer_class` in `tokenizer_config.json` should be changed from `LLaMATokenizer` to `LlamaTokenizer`;
2. The `architectures` in `config.json` should be changed from `LLaMAForCausalLM` to `LlamaForCausalLM`.

### FLAN-T5
The user can obtain the [release model](https://huggingface.co/google/flan-t5-xl) from Huggingface.

## 3. Prepare Dataset
The instruction-following dataset is needed for the finetuning. We select two kinds of Datasets to conduct the finetuning process: general domain dataset and domain specific dataset.

1. General domain dataset: We use the [Alpaca dataset](https://github.com/tatsu-lab/stanford_alpaca) from Stanford University as the general domain dataset to fine-tune the model. This dataset is provided in the form of a JSON file, [alpaca_data.json](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json). In Alpaca, researchers have manually crafted 175 seed tasks to guide `text-davinci-003` in generating 52K instruction data for diverse tasks.

2. Domain-specific dataset: Inspired by Alpaca, we constructed a domain-specific dataset focusing on Business and Intel-related issues. We made minor modifications to the [prompt template](https://github.com/tatsu-lab/stanford_alpaca/blob/main/prompt.txt) to proactively guide Alpaca in generating more Intel and Business related instruction data. The generated data could be find in `intel_domain.json`.

# Finetune

We employ the [LoRA approach](https://arxiv.org/pdf/2106.09685.pdf) to finetune the LLM efficiently, currently, LLaMA and FLAN-T5 are supported for finetuning.

## 1. LLaMA

For LLaMA, use the below command line for finetuning on the Alpaca dataset.

```bash
python finetune.py \
        --model_name_or_path "decapoda-research/llama-7b-hf" \
        --data_path "/path/to/alpaca_data.json" \
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
        --output_dir ./llama_finetuned_model
```

Where the `--dataset_concatenation` argument is a way to vastly accelerate the fine-tuning process through training samples concatenation. With several tokenized sentences concatenated into a longer and concentrated sentence as the training sample instead of having several training samples with different lengths, this way is more efficient due to the parallelism characteristic provided by the more concentrated training samples.

For finetuning in SPR, add `--bf16` argument will further speedup the finetuning process without the loss of model's performance.

## 2. FLAN-T5

For FLAN-T5, use the below command line for finetuning on the Alpaca dataset.

```bash
python finetune_flan_t5.py \
        --model_name_or_path "google/flan-t5-xl/" \
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
        --output_dir ./flan-t5-xl_peft_finetuned_model
```
## 3. Distributed Data Parallel Support

We also supported Distributed Data Parallel finetuning on single node and multi nodes settings. To use Distributed Data Parallel to speedup training, the bash command needs a small adjustment.
<br>
For example, to finetune LLaMA through Distributed Data Parallel training, bash command will look like the following, where
<br>
*`<MASTER_ADDRESS>`* is the address of the master node, it won't be necessary for single node case,
<br>
*`<NUM_PROCESSES_PER_NODE>`* is the desired processes to use in current node, for node with GPU, usually set to number of GPUs in this node, for node without GPU and use CPU for training, it's recommended set to 1,
<br>
*`<NUM_NODES>`* is the number of nodes to use,
<br>
*`<NODE_RANK>`* is the rank of the current node, rank starts from 0 to *`<NUM_NODES>`*`-1`.
<br>
> Also please note that to use CPU for training in each node with multi nodes settings, argument `--no_cuda` is mandatory, and `--xpu_backend ccl` is required if to use ccl as the distributed backend. In multi nodes setting, following command needs to be launched in each node, and all the commands should be the same except for *`<NODE_RANK>`*, which should be integer from 0 to *`<NUM_NODES>`*`-1` assigned to each node.

``` bash
python -m torch.distributed.launch --master_addr=<MASTER_ADDRESS> --nproc_per_node=<NUM_PROCESSES_PER_NODE> --nnodes=<NUM_NODES> --node_rank=<NODE_RANK> \
    finetune.py \
        --model_name_or_path "decapoda-research/llama-7b-hf" \
        --data_path "/path/to/alpaca_data.json" \
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
        --output_dir ./llama_finetuned_model
```
# Chat with the Finetuned Model

Once the model is finetuned, use the below command line to chat with it.
```bash
python generate.py \
        --base_model_path "decapoda-research/llama-7b-hf" \
        --lora_model_path "./llama_finetuned_model" \
        --instructions "What is the capital of China?" "Transform the following sentence into one that shows contrast. The tree is rotten."
```


# Benchmark Result Analysis
With 2K domain knowledge fine-tuning, we can observe a noticeable improvement in the model's ability to explain domain-related questions.

-----
$\color{red}{Prompt:}$ *Tell me the development history of Intel:*

LLaMA origin: Intel was founded in 1968 by Gordon Moore, who was a professor at Stanford University. He was the one who predicted the law of Moore’s law, which says that the number of transistors on a chip doubles every two years.
In 1971, Intel introduced the first microprocessor, the 4004. In 1974, Intel introduced the first microprocessor for personal computers, the 8088. In 1981, Intel introduced the first microprocessor for the PC, the 8086. In 1989, Intel introduced the first Pentium processor. In 1993, Intel introduced the first Pentium Pro processor. In 1997, Intel introduced the first Pentium II processor. In 1999, Intel introduced the first Pentium III processor. In 2003, Intel introduced the first Pentium 4 processor. In 2005, Intel introduced the first Core processor. In 2007, Intel introduced the first Core 2 processor. In 2009, Intel introduced the first Core i7 processor. In 2011, Intel introduced the first Core i5 processor. In 2013, Intel introduced the first Core i3 processor. In 2015, Intel introduced the first Core M processor. In 2016, Intel introduced the first Core i7 processor. In 2017, Intel introduced the first Core i5 processor. In 2018, Intel introduced the first Core i3 processor.

Alpaca: Intel was founded in 1968 by Gordon Moore and Robert Noyce, two former employees of Fairchild Semiconductor. The company’s first product was a semiconductor memory chip called the Intel 1401, which was followed by the Intel 8008 and Intel 8080 microprocessors. In 1989, Intel introduced the world’s first x86-based microcomputer, the Intel 286. In 1998, Intel introduced the Pentium processor, which was the first processor to use the Hyper-Threading technology. In 2006, Intel introduced the Core microarchitecture, which was the first processor to use the Intel 64 instruction set. In 2011, Intel introduced the Ivy Bridge processor, which was the first processor to use the 22nm process. In 2013, Intel introduced the Haswell processor, which was the first processor to use the 22nm Tri-Gate transistor technology. In 2015, Intel introduced the Broadwell processor, which was the first processor to use the 14nm process. In 2017, Intel introduced the Kaby Lake processor, which was the first processor to use the 14nm+ process. In 2018, Intel introduced the Coffee Lake processor, which was the first processor to use the 14nm++ process.

**Ours**: Intel was founded in 1968 by Bob Noyce and Gordon Moore, two pioneers in the semiconductor industry. Intel's first microprocessor, the 4004, was released in 1971. This was the world's first single chip microprocessor and it revolutionized the industry. Intel's first microprocessor for the consumer market, the 8008, was released in 1978. This was the first 8-bit microprocessor and it was used in many early personal computers. In 1981, Intel released the 8086, the first 16-bit microprocessor. This was the basis for the x86 architecture which is still used in modern processors. In 1985, Intel released the 80386, the first 32-bit processor. This was the first processor to feature a 32-bit data bus and it was used in many early PCs. In 1993, Intel released the Pentium processor, which was the first processor to feature a 166 MHz clock speed. This was the first processor to feature a built-in cache and it was used in many early PCs. In 1997, Intel released the Pentium II processor, which was the first processor to feature a 266 MHz clock speed. This was the first processor to feature a built-in cache and it was used in many early PCs. In 1999, Intel released the Pentium III processor, which was the first processor to feature a 333 MHz clock speed. This was the first processor to feature a built-in cache and it was used in many early PCs. In 2000, Intel released the Pentium 4 processor, which was the first processor to feature a 400 MHz clock speed. This was the first processor to feature a built-in cache and it was used in many early PCs. In 2003, Intel released the Pentium 4 processor, which was the first processor to feature a 533 MHz clock speed. This was the first processor to feature a built-in cache and it was used in many early PCs. In 2006, Intel released the Pentium D processor, which was the first dual-core processor. This was the first processor to feature a built-in cache and it was used in many early PCs. In 2008, Intel released the Pentium E processor, which was the first processor to feature a 660 MHz clock speed. This was the first processor to feature a built-in cache and it was used in many early PCs. In 2010, Intel released the Pentium G processor, which was the first processor to feature a 800 MHz clock speed. This was the first processor to feature a built-in cache and it was used in many early PCs. In 2013, Intel released the Pentium 4405U processor, which was the first processor to feature a 1.0 GHz clock speed. This was the first processor to feature a built-in cache and it was used in many early PCs. In 2015, Intel released the Pentium 4405U processor, which was the first processor to feature a 1.1 GHz clock speed. This was the first processor to feature a built-in cache and it was used in many early PCs. In 2017, Intel released the Pentium 4405U processor, which was the first processor to feature a 1.2 GHz clock speed. This was the first processor to feature a built-in cache and it was used in many early PCs. In 2019, Intel released the Pentium 4405U processor, which was the first processor to feature a 1.3 GHz clock speed. This was the first processor to feature a built-in cache and it was used in many early PCs.

$\color{red}{Summary:}$ From this example, we can see that all three models can recognize the history of Intel. However, our model can memorize more key points and provide suitable explanations after fine-tuning on 2K samples.

----
$\color{red}{Prompt:}$ *List the three most powerful personal Intel microchips.*

LLaMA origin: 1. The Intel 4004, the first microprocessor, was introduced in 1971. 2. The Intel 8086, the first 16-bit microprocessor, was introduced in 1978. 3. The Intel Pentium, the first 32-bit microprocessor, was introduced in 1993. 4. The Intel Pentium 4, the first 64-bit microprocessor, was introduced in 2003. 5. The Intel Core 2 Duo, the first dual-core microprocessor, was introduced in 2006.

Alpaca: 1. Core i7-8086K: This is the most powerful Intel microchip released in 2018. It has 6 cores and 12 threads, and it runs at a base frequency of 4.0 GHz. It also has a boost frequency of up to 4.3 GHz. 2. Core i9-9900K: This is the most powerful Intel microchip released in 2019. It has 8 cores and 16 threads, and it runs at a base frequency of 3.6 GHz. It also has a boost frequency of up to 5.0 GHz. 3. Core i9-10980X: This is the most powerful Intel microchip released in 2008. It has 10 cores and 20 threads, and it runs at a base frequency of 3.1 GHz. It also has a boost frequency of up to 3.5 GHz.

**Ours**: The three most powerful personal Intel microchips are the Core i9-9900K, the Core i7-9700K, and the Core i5-9600K. The Core i9-9900K is the most powerful, offering up to 18 cores and 36 threads for high-end gaming and content creation. The Core i7-9700K is a close second, offering up to 8 cores and 16 threads for more mainstream gaming and content creation. The Core i5-9600K is the third most powerful, offering up to 6 cores and 12 threads for more budget-friendly gaming and content creation.

$\color{red}{Summary:}$ In this example, all three models are based on outdated knowledge. Therefore, they do not know the latest product information. Conditioned on the same knowledge base, LLaMA cannot distinguish which microchip performs better, so it lists three products as the answer. Alpaca seems to provide a convincing answer, but the product information about i9-10980X is entirely fabricated. Only our model can give the correct ranking for personal microchips.

----
$\color{red}{Prompt:}$ *Give three suggestion for the Intel marketing strategy and give the reasons.*

LLaMA origin: The marketing strategy of Intel is to focus on the high-end market. The company has a strong brand image and a good reputation in the market. 

Alpaca: 1. Increase the visibility of Intel products by engaging in more marketing campaigns and sponsorships. 2. Develop more innovative products and services to meet the changing needs of customers. 3. Focus on digital marketing to reach a wider audience and increase customer engagement.

**Ours**: Intel should focus on innovation and technology leadership, targeting specific markets with tailored products and services, and leveraging partnerships to expand its reach. Innovation and technology leadership will help Intel stay ahead of the competition and create new opportunities. Targeting specific markets with tailored products and services will allow Intel to better address customer needs and create a more personalized customer experience. Leveraging partnerships will help Intel expand its reach and increase its visibility in the market.

$\color{red}{Summary:}$ Since our fine-tuning dataset contains many business-related instances, we further want to exemplify the generation ability in the business domain. From the given example, we can see that the original LLaMA cannot provide suitable suggestions on the given topic. Although both Alpaca and our model have provided different suggestions, our model offers more convincing explanations for this question.

