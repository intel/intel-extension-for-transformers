
#!/bin/bash
python -u run_clm_no_trainer.py \
    --model /models/llama-7b-hf/ \
    --output_dir ./saved_results \
    --dataset /dataset/pile_10k/ \
    --quantize \
    --approach weight_only \
    --weight_only_algo RTN \
    --weight_only_bits 4 \
    --weight_only_group 32 \
    --weight_only_scheme asym \
    --layer_wise