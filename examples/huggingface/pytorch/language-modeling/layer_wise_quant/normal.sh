
#!/bin/bash
numactl -m 0 -C 0-55 python -u run_clm_no_trainer.py \
    --model decapoda-research/llama-7b-hf \
    --output_dir ./saved_results \
    --dataset /home/hengguo/pile_10k/ \
    --quantize \
    --approach weight_only \
    --weight_only_algo RTN \
    --weight_only_bits 4 \
    --weight_only_group 32 \
    --weight_only_scheme asym \
    --layer_wise \
    # --sq \
    # --alpha 0.5

mprof run -o llama13b_gptq_lwq.dat run_clm_no_trainer.py     --model decapoda-research/llama-13b-hf     --output_dir ./saved_results     --dataset /home/hengguo/pile_10k/     --quantize     --approach weight_only     --weight_only_algo GPTQ     --weight_only_bits 4     --weight_only_group 32     --weight_only_scheme asym     --pad_max_length 128     --layer_wise