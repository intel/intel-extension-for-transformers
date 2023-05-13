Step-by-Step
=========

In this example, we provide the inference benchmarking script `run_gptj.py` for [EleutherAI/gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6B)

>**Note**: The default search algorithm is beam search with num_beams = 4

## Create Environment
```bash
WORK_DIR=$PWD
# Create Environment (conda)
conda create -n llm python=3.9 -y
conda install mkl mkl-include -y
conda install gperftools jemalloc==5.2.1 -c conda-forge -y
pip install -r requirements.txt
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

The fp32 model is [EleutherAI/gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6B), and int8 model has been publiced on [Intel/gpt-j-6B-pytorch-int8-static](https://huggingface.co/Intel/gpt-j-6B-pytorch-int8-static).

### Generate IR
```bash
# fp32 / bf16
python gen_ir.py --model=EleutherAI/gpt-j-6B --dtype=[fp32|bf16] --output_model=<path to ir>

# int8
wget https://huggingface.co/Intel/gpt-j-6B-pytorch-int8-static/resolve/main/pytorch_model.bin -O <path to int8_model.pt>
python gen_ir.py --model=EleutherAI/gpt-j-6B --dtype=int8 --output_model=<path to ir> --pt_file=<path to int8_model.pt>
```
- When the input dtype is fp32 or bf16, the pt file will be automatically saved if it does not exist.
- When the input dtype is int8, the pt file should exist.

### Inference 
We supports inference on multiple sockets with fp32/bf16/int8 Neural Engine model
```bash
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_gptj.py --max-new-tokens 32 --input-tokens 32 --batch-size 1 --ir_path <path to ir>
```
### Advanced Inference
Neural Engine also supports weight compression to `fp8_4e3m`, `fp8_5e2m` and `int8` **only when runing bf16 graph**. If you want to try, please add arg `--weight_type`, like:
```bash
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_gptj.py --max-new-tokens 32 --input-tokens 32 --batch-size 1 --ir_path <path to bf16 ir> --weight_type=fp8_5e2m
```
