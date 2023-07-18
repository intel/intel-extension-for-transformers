Step-by-Step
=========

In this example, we provide the inference benchmarking script `run_llm.py` for [EleutherAI/gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6B), [decapoda-research/llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf), [EleutherAI/gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b) and [databricks/dolly-v2-3b](https://huggingface.co/databricks/dolly-v2-3b) etc.

>**Note**: The default search algorithm is beam search with num_beams = 4

## Create Environment
```bash
# Create Environment (conda)
conda create -n llm python=3.9 -y
conda install mkl mkl-include -y
conda install gperftools jemalloc==5.2.1 -c conda-forge -y
pip install -r requirements.txt

# if you want to run llama model, please install transformers in following version:
pip install git+https://github.com/huggingface/transformers.git@97a3d16a6941294d7d76d24f36f26617d224278e
```

## Environment Variables
```bash
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0
# IOMP
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
# Tcmalloc is a recommended malloc implementation that emphasizes fragmentation avoidance and scalable concurrency support.
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
```
## Performance

The fp32 model is from Hugging Face [EleutherAI/gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6B), [decapoda-research/llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf), [decapoda-research/llama-13b-hf](https://huggingface.co/decapoda-research/llama-13b-hf), [databricks/dolly-v2-3b](https://huggingface.co/databricks/dolly-v2-3b), [EleutherAI/gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b, and gpt-j int8 model has been publiced on [Intel/gpt-j-6B-pytorch-int8-static](https://huggingface.co/Intel/gpt-j-6B-pytorch-int8-static).

### Generate Neural Engine model
```bash
python optimize_llm.py --model=EleutherAI/gpt-j-6B --dtype=(fp32|bf16) --output_model=<path to engine model>

# int8
wget https://huggingface.co/Intel/gpt-j-6B-pytorch-int8-static/resolve/main/pytorch_model.bin -O <path to int8_model.pt>
python optimize_llm.py --model=EleutherAI/gpt-j-6B --dtype=int8 --output_model=<path to ir> --pt_file=<path to int8_model.pt>
```
- When the input dtype is fp32 or bf16, the model will be downloaded if it does not exist.
- When the input dtype is int8, the int8 trace model should exist.

### Inference 
We support inference with FP32/BF16/INT8 Neural Engine model.
```bash
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_llm.py --max-new-tokens 32 --input-tokens 32 --batch-size 1 --model <model name> --model_path <path to engine model>
```

### Advanced Inference
Neural Engine also supports weight compression to `fp8_4e3m`, `fp8_5e2m` and `int8` **only when runing bf16 graph**. If you want to try, please add arg `--weight_type`, like:
```bash
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_llm.py --max-new-tokens 32 --input-tokens 32 --batch-size 1 --model_path <path to bf16 engine model> --model <model name> --weight_type=fp8_5e2m
```
