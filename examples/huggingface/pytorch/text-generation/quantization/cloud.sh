#!/bin/bash
set -x
# config1: 
    # use_quant_input: True; 
    # minmax_lr: 1/200
model_path=$1
cd ~/intel-extension-for-transformers/examples/huggingface/pytorch/text-generation/quantization
CUDA_VISIBLE_DEVICES=0 nohup python run_generation_gpu_woq.py \
    --model $model_path \
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
    --model $model_path \
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
    --model $model_path \
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
    --model $model_path \
    --woq --woq_algo AutoRound  \
    --calib_iters 200  \
    --lr 5e-3 \
    --minmax_lr 1e-2 \
    --output_dir llama3_conf4  \
    --nsamples 512  > test_4.log 2>&1 & 


## iters 1000

cd ~/intel-extension-for-transformers/examples/huggingface/pytorch/text-generation/quantization
CUDA_VISIBLE_DEVICES=4 nohup python run_generation_gpu_woq.py \
    --model $model_path \
    --woq --woq_algo AutoRound  \
    --use_quant_input \
    --calib_iters 1000  \
    --lr 1e-3 \
    --minmax_lr 1e-3 \
    --output_dir llama3_conf5  \
    --nsamples 512  > test_5.log 2>&1 & 

cd ~/intel-extension-for-transformers/examples/huggingface/pytorch/text-generation/quantization
CUDA_VISIBLE_DEVICES=5 nohup python run_generation_gpu_woq.py \
    --model $model_path \
    --woq --woq_algo AutoRound  \
    --use_quant_input \
    --calib_iters 1000  \
    --lr 1e-3 \
    --minmax_lr 2e-3 \
    --output_dir llama3_conf6  \
    --nsamples 512  > test_6.log 2>&1 & 

cd ~/intel-extension-for-transformers/examples/huggingface/pytorch/text-generation/quantization
CUDA_VISIBLE_DEVICES=6 nohup python run_generation_gpu_woq.py \
    --model $model_path \
    --woq --woq_algo AutoRound  \
    --calib_iters 1000  \
    --lr 1e-3 \
    --minmax_lr 1e-3 \
    --output_dir llama3_conf7  \
    --nsamples 512  > test_7.log 2>&1 & 

cd ~/intel-extension-for-transformers/examples/huggingface/pytorch/text-generation/quantization
CUDA_VISIBLE_DEVICES=7 nohup python run_generation_gpu_woq.py \
    --model $model_path \
    --woq --woq_algo AutoRound  \
    --calib_iters 1000  \
    --lr 1e-3 \
    --minmax_lr 2e-3 \
    --output_dir llama3_conf8  \
    --nsamples 512  > test_8.log 2>&1 & 