#!/bin/bash
set -x

# Set environment
export CUBLAS_WORKSPACE_CONFIG=':4096:8'

# Available Models

# Common Large Language Models(LLMs), e.g. OPT, GPT, LLaMA, BLOOM, Dolly, MPT, Falcon, Stable-LM, LaMini-LM, etc.

#cd intel-extension-for-transformers
python examples/huggingface/pytorch/language-modeling/pruning/run_clm_sparsegpt.py \
    --model_name_or_path /PATH/TO/LLM/ \
    --calibration_dataset_name wikitext-2-raw-v1 \
    --do_prune \
    --device=0 \
    --output_dir=/PATH/TO/SAVE/ \
    --target_sparsity 0.5 \
    --pruning_pattern 1x1



