Step-by-Step
============
This document describes the step-by-step instructions to run large language models (LLMs) on 4th Gen Intel® Xeon® Scalable Processor (codenamed Sapphire Rapids) with PyTorch and Intel® Extension for PyTorch.

The scripts `run_clm.py`, `run_mlm.py` and `run_plm.py` provide three quantization approaches respectively (PostTrainingDynamic, PostTrainingStatic, QuantAwareTraining) based on [Intel® Neural Compressor](https://github.com/intel/neural-compressor) and return last token prediction accuracy by `trainer`.

The script `run_clm_no_trainer.py` supports `GPTJ`, `OPT`, `LLaMA`, `BLOOM` quantization and validates last word prediction accuracy with [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness.git) now, and we are adding more models.

# Prerequisite
## 1. Create Environment
```

# Installation
git clone https://github.com/intel/intel-extension-for-transformers.git
cd intel_extension_for_transformers
pip install -r requirements.txt
git submodule update --init --recursive
python setup.py install
cd examples/huggingface/pytorch/language-modeling/quantization
pip install -r requirements.txt

```

# Run

Here is how to run the scripts:

**Causal Language Modeling (CLM)**

`run_clm_no_trainer.py` quantizes the large language models using the dataset [NeelNanda/pile-10k](https://huggingface.co/datasets/NeelNanda/pile-10k) calibration and validates `lambada_openai`, `piqa`, `winogrande`, `hellaswag` and other datasets accuracy provided by lm_eval, an example command is as follows.
### GPT-J-6b

#### Quantization
```bash
# "--sq" is used to enable smooth quant
# "--int8_bf16_mixed" is used to enable int8-bf16 mixed mode for platform that natively supports bf16
python run_clm_no_trainer.py \
    --model EleutherAI/gpt-j-6B \
    --quantize \
    --dataset NeelNanda/pile-10k \
    --sq \
    --alpha 1.0 \
    --output_dir "saved_results" \
    --ipex \
```

#### Accuracy with lm_eval
```bash
# FP32 Accuracy
python run_clm_no_trainer.py \
    --model EleutherAI/gpt-j-6B \
    --accuracy_only \
    --batch_size 112 \
    --tasks "lambada_openai" "lambada_standard"\
    --int8 \
    --ipex \
    --output_dir "saved_results"  # load int8 model
# to validate FP32 model, please remove "--int8" and "--output_dir".
```
### OPT-1.3b/2.7b/6.7b

#### Quantization

```bash
# "--sq" is used to enable smooth quant
# "--int8_bf16_mixed" is used to enable int8-bf16 mixed mode for platform that natively supports bf16
python run_clm_no_trainer.py \
    --model facebook/opt-2.7b \
    --quantize \
    --dataset NeelNanda/pile-10k \
    --sq \
    --alpha 0.5 \
    --ipex \
    --output_dir "saved_results" \
    --int8_bf16_mixed
```

#### Accuracy with lm_eval
```bash
python run_clm_no_trainer.py \
    --model facebook/opt-2.7b \
    --accuracy_only \
    --batch_size 112 \
    --tasks "lambada_openai" "lambada_standard" \
    --int8 \
    --ipex \
    --output_dir "saved_results"  # load int8 model
# to validate FP32 model, please remove "--int8" and "--output_dir".
```
## LLAMA-7b/13b
>Note: LLAMA requires IPEX requirements >= 2.1 to get better accuracy, please source install from [intel_extension_for_pytorch](https://github.com/intel/intel-extension-for-pytorch.git).
#### Quantization

```bash
# "--sq" is used to enable smooth quant
# "--int8_bf16_mixed" is used to enable int8-bf16 mixed mode for platform that natively supports bf16
python run_clm_no_trainer.py \
    --model decapoda-research/llama-7b-hf \
    --quantize \
    --dataset NeelNanda/pile-10k \
    --sq \
    --alpha 0.8 \
    --ipex \
    --output_dir "saved_results" \
    --int8_bf16_mixed
```

#### Accuracy with lm_eval
```bash
python run_clm_no_trainer.py \
    --model decapoda-research/llama-7b-hf \
    --accuracy_only \
    --batch_size 112 \
    --tasks  "lambada_openai" "lambada_standard" \
    --int8 \
    --ipex \
    --output_dir "saved_results"  # load int8 model
# to validate FP32 model, please remove "--int8" and "--output_dir".
```
To do quantization based transformers language-modeling example [`run_clm.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py), please use the following command.
```
python run_clm.py \
    --model_name_or_path EleutherAI/gpt-neo-125M \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --tune \
    --quantization_approach PostTrainingStatic \
    --do_train \
    --do_eval \
    --output_dir ./tmp/clm_output \
    --overwrite_output_dir

```

**Masked Language Modeling (MLM)**

```
python run_mlm.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --tune \
    --quantization_approach PostTrainingStatic \
    --do_train \
    --do_eval \
    --output_dir ./tmp/mlm_output \
    --overwrite_output_dir
```

**Permutation Language Modeling (PLM)**

```
    python run_plm.py \
    --model_name_or_path xlnet-base-cased \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --tune \
    --quantization_approach PostTrainingStatic \
    --do_train \
    --do_eval \
    --output_dir ./tmp/plm_output \
    --overwrite_output_dir

```

