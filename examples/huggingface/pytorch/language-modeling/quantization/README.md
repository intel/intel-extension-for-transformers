Step-by-Step
============
This document describes the step-by-step instructions to run large language models (LLMs) on 4th Gen Intel® Xeon® Scalable Processor (codenamed Sapphire Rapids) with PyTorch and Intel® Extension for PyTorch.

The scripts `run_clm.py`, `run_mlm.py` and `run_plm.py` provide two quantization approaches respectively (PostTrainingDynamic, PostTrainingStatic) based on [Intel® Neural Compressor](https://github.com/intel/neural-compressor).

The script `evaluate_clm.py` supports `GPTJ`, `OPT`, `LLaMA`, `BLOOM` quantization and validates accuracy with [lm_evaluation_harness](https://github.com/EleutherAI/lm-evaluation-harness.git) now, and we are adding more models.

# Prerequisite
## 1. Create Environment
```
WORK_DIR=$PWD
# Create Environment (conda)
conda create -n llm python=3.9 -y
conda install mkl mkl-include -y
conda install gperftools jemalloc==5.2.1 -c conda-forge -y

# Installation
pip install git+https://github.com/intel-innersource/frameworks.ai.nlp-toolkit.intel-nlp-toolkit.git
pip install neural_compressor intel_extension_for_pytorch transformers datasets accelerate

# Setup Environment Variables
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0
# IOMP
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
# Tcmalloc is a recommended malloc implementation that emphasizes fragmentation avoidance and scalable concurrency support.
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
```

# Run
## 1. Quantization

Here is how to run the scripts:

**Causal Language Modeling (CLM)**

`evaluate_clm.py` quantizes the large language models using the dataset [NeelNanda/pile-10k](https://huggingface.co/datasets/NeelNanda/pile-10k) calibration and validates `lambada_openai`, `piqa`, `winogrande`, `hellaswag` and other datasets accuracy provided by lm_evaluation_harness, an example command is as follows.
```bash
# "--sq" is used to enable smooth quant
# "--int8_bf16_mixed" is used to enable int8-bf16 mixed mode for platform that natively supports bf16
python evaluate_clm.py \
    --model EleutherAI/gpt-j-6B \
    --quantize \
    --dataset lambada \
    --sq \
    --alpha 0.7 \
    --output_dir "saved_results"
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
## 2. Accuracy
```bash
# FP32 Accuracy
python evaluate_clm.py \
    --model EleutherAI/gpt-j-6B \
    --accuracy_only \
    --batch_size 56

# BF16 Accuracy
python evaluate_clm.py \
    --model EleutherAI/gpt-j-6B \
    --accuracy_only \
    --ipex_bf16 \
    --batch_size 56

# INT8 Accuracy
python evaluate_clm.py \
    --model EleutherAI/gpt-j-6B \
    --accuracy_only \
    --int8 \
    --batch_size 56
```

## 3. Validated Model List

|Type|Pretrained model|PostTrainingDynamic | PostTrainingStatic | QuantizationAwareTraining
|---|------------------------------------|---|---|---
|CLM|EleutherAI/gpt-neo-125M| ✅| ✅| Stay tuned
|CLM|abeja/gpt-neox-japanese-2.7b| ✅| ✅| Stay tuned
|CLM|EleutherAI/gpt-j-6B| ✅| ✅| Stay tuned
|CLM|bigscience/bloom-560m| ✅| ✅| Stay tuned
|MLM|bert-base-uncased| ✅| ✅| Stay tuned
|PLM|xlnet-base-cased| ✅| ✅| Stay tuned

## 3. Bash Command

```
bash run_tuning.sh  --topology=topology
```

```
bash run_benchmark.sh --topology=topology --mode=benchmark
```
> NOTE
>
> topology should be one of: {"gpt_neo_clm_static", "gpt_neo_clm_dynamic", "gptj_clm_static", "gptj_clm_dynamic", "gpt_neox_clm_static", "gpt_neox_clm_dynamic", "bert_mlm_static", "bert_mlm_dynamic", "xlnet_plm_static", "xlnet_plm_dynamic", "reformer_crime_and_punishment_static", "ctrl_wikitext_static", "bloom_clm_static"}
