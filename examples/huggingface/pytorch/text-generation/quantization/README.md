# Step-by-Step
We provide the inference benchmarking script `run_generation.py` for [EleutherAI/gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6B),  [decapoda-research/llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf), [decapoda-research/llama-13b-hf](https://huggingface.co/decapoda-research/llama-13b-hf), [databricks/dolly-v2-3b](https://huggingface.co/databricks/dolly-v2-3b), [bigscience/bloom-7b1](https://huggingface.co/bigscience/bloom-7b1), [facebook/opt-1.3b](https://huggingface.co/facebook/opt-1.3b), [facebook/opt-2.7b](https://huggingface.co/facebook/opt-2.7b), [facebook/opt-6.7b](https://huggingface.co/facebook/opt-6.7b), [mosaicml/mpt-7b-chat](https://huggingface.co/mosaicml/mpt-7b-chat), [Intel/neural-chat-7b-v1-1](https://huggingface.co/Intel/neural-chat-7b-v1-1), more models are working in progress.

>**Note**: The default search algorithm is beam search with num_beams = 4, if you'd like to use greedy search for comparison, add "--greedy" in args.


# Prerequisite​
## 1. Create Environment​
Pytorch and Intel-extension-for-pytorch version 2.1 are required, the dependent packages are listed in requirements, we recommend create environment as the following steps.

```bash
conda create -n llm python=3.9 -
conda activate llm
bash build_env.sh
git clone https://github.com/intel/intel-extension-for-transformers.git
cd intel-extension-for-transformers
pip install -r requirements.txt
python setup.py install
```
> Note:
> Disable semi-compiler to avoid accuracy regression for mpt and neural-chat-v1-1 models, other > models don't need it.
> `export _DNNL_DISABLE_COMPILER_BACKEND=1`

> Note: If `ImportError: /lib64/libstdc++.so.6: version ``GLIBCXX_3.4.29`` not found` error raised when import intel-extension-for-pytorch, it is due to the high gcc library request, there is the solution to find the correct version.
> ```bash
> find $CONDA_PREFIX | grep libstdc++.so.6
> export LD_PRELOAD=<the path of libstdc++.so.6>:${LD_PRELOAD}
> ```



# Run
We support compression technologies such as `MixedPrecision`, `SmoothQuant` and `WeightOnlyQuant` with `RTN/AWQ/TEQ/GPTQ` algorithms, `BitsAndBytes` based transformers also works, the followings are command to show how to use it.

## 1. Performance
``` bash
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export LD_PRELOAD=${CONDA_PREFIX}/lib/libiomp5.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
# fp32
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_generation.py \
    --model EleutherAI/gpt-j-6b \
    --benchmark
# mixedprecision
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_generation.py \
    --model EleutherAI/gpt-j-6b \
    --mixed_precision \
    --benchmark
# smoothquant
# --int8_bf16_mixed is used for int8 mixed bfloat16 precision.
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_generation.py \
    --model EleutherAI/gpt-j-6b \
    --sq \
    --alpha 1.0 \
    --int8 \
    --benchmark
# weightonlyquant
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_generation.py \
    --model EleutherAI/gpt-j-6b \
    --woq \
    --benchmark
# bitsandbytes
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_generation.py \
    --model EleutherAI/gpt-j-6b \
    --bitsandbytes \
    --benchmark

```

## 2. Accuracy
```bash
# fp32
python run_generation.py \
    --model EleutherAI/gpt-j-6b \
    --accuracy \
    --tasks "lambada_openai"
# mixedprecision
python run_generation.py \
    --model EleutherAI/gpt-j-6b \
    --mixed_precision \
    --accuracy \
    --tasks "lambada_openai"
# smoothquant
# --int8_bf16_mixed is used for int8 mixed bfloat16 precision.
python run_generation.py \
    --model EleutherAI/gpt-j-6b \
    --sq \
    --alpha 1.0 \
    --int8 \
    --accuracy \
    --tasks "lambada_openai"
# weightonlyquant
python run_generation.py \
    --model EleutherAI/gpt-j-6b \
    --woq \
    --accuracy \
    --tasks "lambada_openai"
# bitsandbytes
python run_generation.py \
    --model EleutherAI/gpt-j-6b \
    --bitsandbytes \
    --accuracy \
    --tasks "lambada_openai"
```
