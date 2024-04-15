1. generate llama3 INT4 IPEX-GPU format
```python
python run_generation_gpu_woq.py --model /data5/llama_random_8b_hf --woq --woq_algo AutoRound  --calib_iter 1  --output_dir llama3_all_int4
```
 - Quantization on NV GPU   
    ITREX branch: llama3_xpu_int4   
    ```python
    python run_generation_gpu_woq.py --model /data5/llama_random_8b_hf --woq --woq_algo AutoRound  --calib_iter 1    --use_quant_input --output_dir llama3_all_int4
    ```
 - load XPU   
    ITREX branch: penghuic/add_lm_head   
    ```python
    python run_generation_gpu_woq.py --model llama3_all_int4 --accuracy
    ```

llama2
python run_generation_gpu_woq.py --model  /models/Llama-2-7b-chat-hf  --woq --woq_algo AutoRound  --calib_iter 1000  --use_quant_input --output_dir llama2_chat_all_int4

tiny llama2
python run_generation_gpu_woq.py --model yujiepan/llama-2-tiny-random  --woq --woq_algo AutoRound  --calib_iter 1  --output_dir llama3_all_int4  --use_quant_input
