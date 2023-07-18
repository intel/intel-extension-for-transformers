Step-by-Step
============
This document describes the step-by-step instructions to run large language models (LLMs) on 4th Gen Intel® Xeon® Scalable Processor (codenamed Sapphire Rapids) with PyTorch and Intel® Extension for PyTorch.

The scripts `run_clm.py`, `run_mlm.py` and `run_plm.py` provide three quantization approaches respectively (PostTrainingDynamic, PostTrainingStatic, QuantAwareTraining) based on [Intel® Neural Compressor](https://github.com/intel/neural-compressor) and return last token prediction accuracy by `trainer`.

The script `run_clm_no_trainer.py` supports `GPTJ`, `OPT`, `LLaMA`, `BLOOM`, `MPT` and `Falcon` quantization and validates last word prediction accuracy with [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness.git) now, and we are adding more models.

# Prerequisite
## 1. Create Environment
```
# Installation
git clone https://github.com/intel/intel-extension-for-transformers.git itrex
cd itrex
pip install -v .
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
    --accuracy \
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
    --accuracy \
    --batch_size 112 \
    --tasks "lambada_openai" "lambada_standard" \
    --int8 \
    --ipex \
    --output_dir "saved_results"  # load int8 model
# to validate FP32 model, please remove "--int8" and "--output_dir".
```
### LLAMA-7b/13b/30b
>Note: LLAMA requires IPEX requirements >= 2.1 to get better accuracy, please source install from [intel_extension_for_pytorch](https://github.com/intel/intel-extension-for-pytorch.git).
#### Quantization

```bash
# "--sq" is used to enable smooth quant
# "--int8_bf16_mixed" is used to enable int8-bf16 mixed mode for platform that natively supports bf16
python run_clm_no_trainer.py \
    --model decapoda-research/llama-7b-hf \
    --quantize \
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

### MPT-7b-chat
#### Quantization
`mosaicml/mpt-7b` has been updated frequently, and has not yet been integrated into `transformers`, so we fixed a commit number `68e1a8e0ebb9b30f3c45c1ef6195980f29063ae2` as local folder to enable it.
```bash
# "--sq" is used to enable smooth quant
# "--int8_bf16_mixed" is used to enable int8-bf16 mixed mode for platform that natively supports bf16
python run_clm_no_trainer.py \
    --model mosaicml/mpt-7b-chat \
    --quantize \
    --sq \
    --alpha 0.85 \
    --ipex \
    --output_dir "saved_results"
```

#### Accuracy with lm_eval
```bash
python run_clm_no_trainer.py \
    --model mosaicml/mpt-7b-chat \
    --accuracy_only \
    --batch_size 112 \
    --tasks  "lambada_openai" \
    --int8 \
    --ipex \
    --output_dir "saved_results"  # load int8 model
# to validate FP32 model, please remove "--int8" and "--output_dir".
```
### Falcon-7b-instruct
#### Quantization
`tiiuae/falcon-7b-instruct` has been updated frequently, and has not yet been integrated into `transformers`, so we fixed a commit number `c7f670a03d987254220f343c6b026ea0c5147185` as local folder to enable it.
```bash
# "--sq" is used to enable smooth quant
# "--int8_bf16_mixed" is used to enable int8-bf16 mixed mode for platform that natively supports bf16
python run_clm_no_trainer.py \
    --model tiiuae/falcon-7b-instruct \
    --quantize \
    --sq \
    --alpha 0.7 \
    --output_dir "saved_results"
```

#### Accuracy with lm_eval
```bash
python run_clm_no_trainer.py \
    --model tiiuae/falcon-7b-instruct \
    --accuracy_only \
    --batch_size 112 \
    --tasks  "lambada_openai" \
    --int8 \
    --ipex \
    --output_dir "saved_results"  # load int8 model
# to validate FP32 model, please remove "--int8" and "--output_dir".
```


To do quantization based transformers language-modeling example [`run_clm.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py), please use the following command.
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

 
