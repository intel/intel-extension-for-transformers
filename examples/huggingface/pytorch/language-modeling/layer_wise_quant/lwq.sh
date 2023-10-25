
#!/bin/bash
python -u run_clm_no_trainer.py \
    --model decapoda-research/llama-13b-hf \
    --output_dir ./saved_results \
    --dataset /home/hengguo/pile_10k/ \
    --quantize \
    --approach weight_only \
    --weight_only_algo GPTQ \
    --weight_only_bits 4 \
    --weight_only_group 32 \
    --weight_only_scheme asym \
    --pad_max_length 128 \
    --layer_wise
