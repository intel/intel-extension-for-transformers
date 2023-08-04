# "--gptq" is used to enable gptq algorithm
python examples/huggingface/pytorch/language-modeling/quantization/run_clm_no_trainer.py \
    --model facebook/opt-125m \
    --quantize \
    --approach weight_only \
    --weight_only_algo GPTQ \
    --weight_only_bits 4 \
    --weight_only_group 128 \
    --output_dir "test_models" \
    --tasks lambada_openai \
    --weight_only_scheme asym