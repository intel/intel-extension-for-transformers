git+https://github.com/huggingface/optimum-intel.git@74912c0caa7dfb8979f59cbc94b57f5d6a448c30# Step-by-Step
We provide the inference benchmarking script `run_generation.py` for large language models, The following are the models we validated, more models are working in progress.

|Validated models| Smoothquant alpha |
|---| ---|
|[EleutherAI/gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6B)| 1.0 |
|[decapoda-research/llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf)| 0.7 |
|[decapoda-research/llama-13b-hf](https://huggingface.co/decapoda-research/llama-13b-hf)| 0.8 |
|[lmsys/vicuna-7b-v1.3](https://huggingface.co/lmsys/vicuna-7b-v1.3)| 0.7 |
|[meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)| 1.0 |
|[meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)| 1.0 |
|[databricks/dolly-v2-3b](https://huggingface.co/databricks/dolly-v2-)| 0.5 |
|[bigscience/bloom-560m](https://huggingface.co/bigscience/bloom-560m)| 0.5 |
|[bigscience/bloom-1b7](https://huggingface.co/bigscience/bloom-1b7)| 0.5 |
|[bigscience/bloom-7b1](https://huggingface.co/bigscience/bloom-7b1)| 0.5 |
|[bigscience/bloomz-560m](https://huggingface.co/bigscience/bloomz-560m)| 0.5 |
|[bigscience/bloomz-1b7](https://huggingface.co/bigscience/bloomz-1b7)| 0.5 |
|[bigscience/bloomz-7b1](https://huggingface.co/bigscience/bloomz-7b1)| 0.5 |
|[facebook/opt-1.3b](https://huggingface.co/facebook/opt-1.3b)| 0.5 |
|[facebook/opt-2.7b](https://huggingface.co/facebook/opt-2.7b)| 0.5 |
|[facebook/opt-6.7b](https://huggingface.co/facebook/opt-6.7b)| 0.5 |
|[mosaicml/mpt-7b-chat](https://huggingface.co/mosaicml/mpt-7b-chat)| 1.0 |
|[Intel/neural-chat-7b-v1-1](https://huggingface.co/Intel/neural-chat-7b-v1-1)| 1.0 |

>**Note**: The default search algorithm is beam search with num_beams = 4, if you'd like to use greedy search for comparison, add "--greedy" in args.


# Prerequisite​
## 1. Create Environment​
Pytorch and Intel-extension-for-pytorch version 2.1 are required, the dependent packages are listed in requirements, we recommend create environment as the following steps.

```bash
pip install -r requirements.txt
```

> Note: If `ImportError: /lib64/libstdc++.so.6: version ``GLIBCXX_3.4.29`` not found` error raised when import intel-extension-for-pytorch, it is due to the high gcc library request, there is the solution to find the correct version.
> ```bash
> find $CONDA_PREFIX | grep libstdc++.so.6
> export LD_PRELOAD=<the path of libstdc++.so.6>:${LD_PRELOAD}
> ```


# Run
We support compression technologies such as `MixedPrecision`, `SmoothQuant` and `WeightOnlyQuant` with `RTN/AWQ/TEQ` algorithms and `BitsandBytes`, `load_in_4bit` and `load_in_8bit` work on CPU device are provided, the followings are command to show how to use it.

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
# [alternative] --int8 is used for int8 only, --int8_bf16_mixed is used for int8 mixed bfloat16 precision.
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
# load_in_4bit
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_generation.py \
    --model EleutherAI/gpt-j-6b \
    --load_in_4bit True \
    --benchmark
# load_in_8bit
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_generation.py \
    --model EleutherAI/gpt-j-6b \
    --load_in_8bit True \
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
# [alternative] --int8 is used for int8 only, --int8_bf16_mixed is used for int8 mixed bfloat16 precision.
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
# load_in_4bit
python run_generation.py \
    --model EleutherAI/gpt-j-6b \
    --load_in_4bit True \
    --accuracy \
    --tasks "lambada_openai"
# load_in_8bit
python run_generation.py \
    --model EleutherAI/gpt-j-6b \
    --load_in_8bit True \
    --accuracy \
    --tasks "lambada_openai"

```
