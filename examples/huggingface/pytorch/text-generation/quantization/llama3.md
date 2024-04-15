1. generate INT4 IPEX-GPU format
```python
python run_generation_gpu_woq.py --model /models/llama3 --woq --woq_algo AutoRound  --calib_iter 1  --nsamples 2 --output_dir llama3_all_int4
```