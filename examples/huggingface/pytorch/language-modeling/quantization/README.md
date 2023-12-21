Step-by-Step
============
This document describes the step-by-step instructions to run large language models (LLMs) on 4th Gen Intel® Xeon® Scalable Processor (codenamed Sapphire Rapids) with PyTorch and Intel® Extension for PyTorch.

The scripts `run_clm.py`, `run_mlm.py` and `run_plm.py` provide three quantization approaches respectively (PostTrainingDynamic, PostTrainingStatic, QuantAwareTraining) based on [Intel® Neural Compressor](https://github.com/intel/neural-compressor) and return last token prediction accuracy by `trainer`.

The large language model quantization is moved to [text-generation](../../text-generation/quantization/) now.

# Prerequisite
## 1. Create Environment
```
# Installation
git clone https://github.com/intel/intel-extension-for-transformers.git itrex
cd itrex
pip install -r requirements.txt
pip install -v .
cd examples/huggingface/pytorch/language-modeling/quantization
pip install -r requirements.txt
pip install transformers==4.34.1
```
>**Note**: Please use transformers no higher than 4.34.1

# Run

Here is how to run the scripts:


**Causal Language Modeling (CLM)**
```bash
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

```bash
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

```bash
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

[1]. Elias, Frantar, et al. "GPTQ: Accurate Post-training Compression for Generative Pretrained Transformers." arXiv preprint arXiv:2210.17323 (2023).
[2]. Lin, Ji, et al. "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." arXiv preprint arXiv:2306.00978 (2023).
