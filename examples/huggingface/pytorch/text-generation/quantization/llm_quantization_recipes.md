# Step-by-Step recipes for LLM quantization

This document describes the step-by-step instructions to run large language models (LLMs) on 4th Gen Intel® Xeon® Scalable Processor (codenamed Sapphire Rapids) with [PyTorch](https://pytorch.org/) and [Intel® Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch).

The scripts [run_generation.py](./run_generation.py) provide two quantization approaches respectively (SmoothQuant, Weight-Only Quantization) based on [Intel® Neural Compressor](https://github.com/intel/neural-compressor) and return last word prediction accuracy by [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/master).

# Validated Models

|                       Model Name                       |
| :----------------------------------------------------: |
|       [EleutherAI/gpt-j-6b](#eleutheraigpt-j-6b)       |
|         [facebook/opt-1.3b](#facebookopt-13b)          |
|          [facebook/opt-30b](#facebookopt-30b)          |
|  [meta-llama/Llama-2-7b-hf](#meta-llamallama-2-7b-hf)  |
| [meta-llama/Llama-2-13b-hf](#meta-llamallama-2-13b-hf) |
| [meta-llama/Llama-2-70b-hf](#meta-llamallama-2-70b-hf) |
|         [tiiuae/falcon-40b](#tiiuaefalcon-40b)         |

# Prerequisite

```bash
# Installation
git clone https://github.com/intel/intel-extension-for-transformers.git

# install ITREX
cd intel-extension-for-transformers
git checkout a4aba8ddb07c9b744b6ac106502ec059e0c47960
pip install -r requirements.txt
pip install -v .

# install requirements
cd examples/huggingface/pytorch/text-generation/quantization
pip install -r requirements.txt
pip install neural-compressor==2.4.1
pip install transformers==4.32.0
pip install torch==2.1.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install intel-extension-for-pytorch==2.1.100
pip uninstall lm_eval -y
pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git@cc9778fbe4fa1a709be2abed9deb6180fd40e7e2
```

# Run Quantization and evaluate INT8 accuracy

## EleutherAI/gpt-j-6b

### SmoothQuant

```bash
python run_generation.py \
    --model EleutherAI/gpt-j-6b \
    --output_dir ./saved_results \
    --trust_remote_code True \
    --fallback_add \
    --tasks lambada_openai \
    --int8 --sq --accuracy \
    --batch_size 1 \
    --alpha 0.85
```

### Weight-Only Quantization

```bash
python run_generation.py \
    --model EleutherAI/gpt-j-6b \
    --output_dir ./saved_results \
    --woq \
    --accuracy
```

## facebook/opt-1.3b

### SmoothQuant

```bash
python run_generation.py \
    --model facebook/opt-1.3b \
    --output_dir ./saved_results \
    --trust_remote_code True \
    --tasks lambada_openai \
    --int8 --sq --accuracy \
    --batch_size 1 \
    --alpha 0.9
```

### Weight-Only Quantization

```bash
python run_generation.py \
    --model facebook/opt-1.3b \
    --output_dir ./saved_results \
    --woq \
    --accuracy
```

## facebook/opt-30b

### SmoothQuant

```bash
python run_generation.py \
    --model facebook/opt-30b \
    --output_dir ./saved_results \
    --trust_remote_code True \
    --tasks lambada_openai \
    --int8 --sq --accuracy \
    --batch_size 1 \
    --alpha 0.5
```

### Weight-Only Quantization

```bash
python run_generation.py \
    --model facebook/opt-30b \
    --output_dir ./saved_results \
    --woq \
    --accuracy
```

## meta-llama/Llama-2-7b-hf

### SmoothQuant

```bash
python run_generation.py \
    --model meta-llama/Llama-2-7b-hf \
    --output_dir ./saved_results \
    --trust_remote_code True \
    --calib_len 2048 \
    --fallback_add \
    --calib_shuffle False \
    --tasks lambada_openai \
    --int8 --sq --accuracy \
    --batch_size 1 \
    --recipes "{'smooth_quant': True, 'smooth_quant_args': {'alpha': 'auto', 'folding': False, 'default_alpha': 0.8, 'auto_alpha_args': {'alpha_min': 0.8, 'alpha_max': 0.99, 'alpha_step': 0.01, 'shared_criterion': 'mean'}}}"
```

### Weight-Only Quantization

```bash
python run_generation.py \
    --model meta-llama/Llama-2-7b-hf \
    --output_dir ./saved_results \
    --woq \
    --accuracy
```

## meta-llama/Llama-2-13b-hf

### SmoothQuant

```bash
python run_generation.py \
    --model meta-llama/Llama-2-13b-hf \
    --output_dir ./saved_results \
    --trust_remote_code True \
    --calib_len 1024 \
    --fallback_add \
    --calib_padding \
    --tasks lambada_openai \
    --int8 --sq --accuracy \
    --batch_size 1 \
    --recipes "{'smooth_quant': True, 'smooth_quant_args': {'alpha': 'auto', 'folding': False, 'default_alpha': 0.8, 'auto_alpha_args': {'alpha_min': 0.75, 'alpha_max': 0.99, 'alpha_step': 0.01, 'shared_criterion': 'max'}}}"
```

### Weight-Only Quantization

```bash
python run_generation.py \
    --model meta-llama/Llama-2-13b-hf \
    --output_dir ./saved_results \
    --woq \
    --accuracy
```

## meta-llama/Llama-2-70b-hf

### SmoothQuant

```bash
python run_generation.py \
    --model meta-llama/Llama-2-70b-hf \
    --output_dir ./saved_results \
    --trust_remote_code True \
    --tasks lambada_openai \
    --int8 --sq --accuracy \
    --batch_size 1 \
    --alpha 0.8
```

### Weight-Only Quantization

```bash
python run_generation.py \
    --model meta-llama/Llama-2-70b-hf \
    --output_dir ./saved_results \
    --woq \
    --accuracy
```

## tiiuae/falcon-40b

```bash
pip install transformers==4.33.3 # for tiiuae/falcon-40b
```

### SmoothQuant

```bash
python run_generation.py \
    --model tiiuae/falcon-40b \
    --output_dir ./saved_results \
    --trust_remote_code True \
    --tasks lambada_openai \
    --int8 --sq --accuracy \
    --batch_size 1 \
    --alpha 0.9
```

### Weight-Only Quantization

```bash
python run_generation.py \
    --model tiiuae/falcon-40b \
    --output_dir ./saved_results \
    --woq \
    --accuracy
```
