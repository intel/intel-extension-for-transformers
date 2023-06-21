Chat with the NeuralChat
============

This example demonstrates how to use the fine-tuned model to chat with NeuralChat. To get the fine-tuned model, you can refer to the [fine-tuning](../fine_tuning/README.md) part.

# Prerequisite​

## 1. Environment​
Recommend python 3.9 or higher version.
```shell
pip install -r requirements.txt
```

# Inference

Take t5 as example, and you could extend it to other models.

For FLAN-T5, use the below command line to chat with it. Take FLAN-T5 as example, and you could extend it to other models.

```bash
python generate.py \
        --base_model_path "google/flan-t5-xl" \
        --peft_model_path  "./flan-t5-xl_peft_finetuned_model" \
        --instructions "Transform the following sentence into one that shows contrast. The tree is rotten."
```

For Llama, use the below command line to chat with it. 
Add option "--use_slow_tokenizer" when using latest transformers if you met failure in llama fast tokenizer  
for llama, The `tokenizer_class` in `tokenizer_config.json` should be changed from `LLaMATokenizer` to `LlamaTokenizer`

```bash
python generate.py \
        --base_model_path "decapoda-research/llama-7b-hf" \
        --peft_model_path "./llama_peft_finetuned_model" \
        --use_slow_tokenizer \
        --instructions "Transform the following sentence into one that shows contrast. The tree is rotten."
```

For [MPT](https://huggingface.co/mosaicml/mpt-7b ). it uses gpt-neox-20b tokenizer, so you need to define it in command line explicitly.This model also requires that trust_remote_code=True be passed to the from_pretrained method. This is because we use a custom MPT model architecture that is not yet part of the Hugging Face transformers package.

```bash
python generate.py \
        --base_model_path "mosaicml/mpt-7b" \
        --peft_model_path "./mpt_peft_finetuned_model" \
        --tokenizer_name "EleutherAI/gpt-neox-20b" \
        --trust_remote_code \
        --instructions "Transform the following sentence into one that shows contrast. The tree is rotten."
```
