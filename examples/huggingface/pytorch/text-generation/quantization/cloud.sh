#!/bin/bash
set -x
# config1: 
    # use_quant_input: True; 
    # minmax_lr: 1/200
cd ~/intel-extension-for-transformers/examples/huggingface/pytorch/text-generation/quantization
CUDA_VISIBLE_DEVICES=0 nohup python run_generation_gpu_woq.py \
    --model  ~/llama3_8b_hf \
    --woq --woq_algo AutoRound  \
    --use_quant_input \
    --calib_iters 200  \
    --lr 5e-3 \
    --minmax_lr 5e-3 \
    --output_dir llama3_conf1  \
    --nsamples 512  > test_1.log 2>&1 & 
# config1: 
    # use_quant_input: True; 
    # minmax_lr: 2/200
cd ~/intel-extension-for-transformers/examples/huggingface/pytorch/text-generation/quantization
CUDA_VISIBLE_DEVICES=1 nohup python run_generation_gpu_woq.py \
    --model  ~/llama3_8b_hf \
    --woq --woq_algo AutoRound  \
    --use_quant_input \
    --calib_iters 200  \
    --lr 5e-3 \
    --minmax_lr 1e-2 \
    --output_dir llama3_conf2  \
    --nsamples 512  > test_2.log 2>&1 & 
# config1: 
    # use_quant_input: False; 
    # minmax_lr: 1/200
cd ~/intel-extension-for-transformers/examples/huggingface/pytorch/text-generation/quantization
CUDA_VISIBLE_DEVICES=2 nohup python run_generation_gpu_woq.py \
    --model  ~/llama3_8b_hf \
    --woq --woq_algo AutoRound  \
    --calib_iters 200  \
    --lr 5e-3 \
    --minmax_lr 5e-3 \
    --output_dir llama3_conf3  \
    --nsamples 512  > test_3.log 2>&1 & 
# config1: 
    # use_quant_input: False; 
    # minmax_lr: 2/200
cd ~/intel-extension-for-transformers/examples/huggingface/pytorch/text-generation/quantization
CUDA_VISIBLE_DEVICES=3 nohup python run_generation_gpu_woq.py \
    --model  ~/llama3_8b_hf \
    --woq --woq_algo AutoRound  \
    --calib_iters 200  \
    --lr 5e-3 \
    --minmax_lr 1e-2 \
    --output_dir llama3_conf4  \
    --nsamples 512  > test_4.log 2>&1 & 

CUDA_VISIBLE_DEVICES=4 lm_eval --model hf \
    --model_args pretrained=~/llama3_8b_hf,dtype="float16"\
    --tasks lambada_openai,hellaswag,piqa,winogrande,truthfulqa_mc1,openbookqa,boolq,rte,arc_easy,arc_challenge\
    --device cuda:0 \
    --batch_size 16  > test_accu_fp16.log 2>&1 & 
