# QLoRA on CPU

1. [Introduction](#introduction)
2. [Examples](#examples)

    2.1. [Python API](#python-api)

    2.2. [Neural Chat Example](#neural-chat-example)

## Introduction
[QLoRA](https://arxiv.org/abs/2305.14314) is an efficient finetuning approach that reduces memory usage of Large Language Models (LLMs) finetuning, it backpropagates gradients through a frozen, quantized LLMs into Low Rank Adapters~(LoRA). Currently it only supports finetuning on CUDA devices, we have developed necessary API to support QLoRA on CPU device, where 4-bit NormalFloat (NF4), Float4 (FP4), INT4 and INT8 are supported data type for LLMs quantization.

## Examples

### Python API

```python
import torch
from intel_extension_for_transformers.transformers.modeling import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    'decapoda-research/llama-7b-hf',
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,
    use_neural_speed=False
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
model = prepare_model_for_kbit_training(
    model, use_gradient_checkpointing=True
)
model.gradient_checkpointing_enable()
peft_config = LoraConfig(
    r=8,
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, peft_config)
```

### Neural Chat Example

To use QLoRA on Neural Chat with CPU device, just add `--qlora` argument to the normal [Neural Chat Fine-tuning Example](https://github.com/intel/intel-extension-for-transformers/tree/main/intel_extension_for_transformers/neural_chat/examples/finetuning/instruction), for example, as below.

```bash
python finetune_clm.py \
        --model_name_or_path "meta-llama/Llama-2-7b" \
        --bf16 True \
        --dataset_name /path/to/alpaca_data.json \
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
        --qlora
        --max_train_samples 500
```
