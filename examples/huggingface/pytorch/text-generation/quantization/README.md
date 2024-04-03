# Step-by-Step
We provide the inference benchmarking script `run_generation.py` for large language models, The following are the models we validated, more models are working in progress.

# Quantization for CPU device

>**Note**: 
> 1.  default search algorithm is beam search with num_beams = 4.
> 2. Model type "gptj", "opt", "llama" and "falcon" will default use [ipex.optimize_transformers](https://github.com/intel/intel-extension-for-pytorch/blob/339bd251841e153ad9c34e1033ab8b2d936a1781/docs/tutorials/llm/llm_optimize_transformers.md) to accelerate the inference, but "llama" requests transformers version lower than 4.36.0, "falcon" requests transformers version lower than 4.33.3 for SmoothQuant.
## Prerequisite​
### Create Environment​
Pytorch and Intel-extension-for-pytorch version 2.1 are required, python version requests equal or higher than 3.9 due to [text evaluation library](https://github.com/EleutherAI/lm-evaluation-harness/tree/master) limitation, the dependent packages are listed in requirements, we recommend create environment as the following steps.

```bash
pip install -r requirements.txt
```

> Note: If `ImportError: /lib64/libstdc++.so.6: version ``GLIBCXX_3.4.29`` not found` error raised when import intel-extension-for-pytorch, it is due to the high gcc library request, there is the solution to find the correct version.
> ```bash
> find $CONDA_PREFIX | grep libstdc++.so.6
> export LD_PRELOAD=<the path of libstdc++.so.6>:${LD_PRELOAD}
> ```


## Run
We provide compression technologies such as `MixedPrecision`, `SmoothQuant` and `WeightOnlyQuant` with `Rtn/Awq/Teq/GPTQ/AutoRound` algorithms and `BitsandBytes`, `load_in_4bit` and `load_in_8bit` work on CPU device, and also support `PEFT` optimized model compression, the followings are command to show how to use it.

### 1. Performance
``` bash
# Please use "--peft_model_id" to replace "--model" if the peft model is used.
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
# --fallback_add option could be enabled to fallback all add-ops to FP32.
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
# restore the model optimized with smoothquant
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_generation.py \
    --model EleutherAI/gpt-j-6b \
    --output_dir saved_results \
    --int8 \
    --restore \
    --benchmark \
    --tasks "lambada_openai"

```

### 2. Accuracy
```bash
# Please use "--peft_model_id" to replace "--model" if the peft model is used.
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
    --woq_weight_dtype "nf4" \
    --accuracy \
    --tasks "lambada_openai"
# load_in_4bit
python run_generation.py \
    --model EleutherAI/gpt-j-6b \
    --load_in_4bit \
    --accuracy \
    --tasks "lambada_openai"
# load_in_8bit
python run_generation.py \
    --model EleutherAI/gpt-j-6b \
    --load_in_8bit \
    --accuracy \
    --tasks "lambada_openai"
# restore the model optimized with smoothquant
python run_generation.py \
    --model EleutherAI/gpt-j-6b \
    --output_dir saved_results \
    --int8 \
    --restore \
    --accuracy \
    --tasks "lambada_openai"

```

# Weight Only Quantization for GPU device
>**Note**: 
> 1.  default search algorithm is beam search with num_beams = 1.
> 2. [ipex.optimize_transformers](https://github.com/intel/intel-extension-for-pytorch/blob/v2.1.10%2Bxpu/docs/tutorials/llm/llm_optimize_transformers.md) Support for the optimized inference of model types "gptj," "mistral," "qwen," and "llama" to achieve high performance and accuracy. Ensure accurate inference for other model types as well.
> 3. We provide compression technologies `WeightOnlyQuant` with `Rtn/GPTQ/AutoRound` algorithms and `load_in_4bit` and `load_in_8bit` work on intel GPU device.

## Prerequisite​
### Dependencies
Intel-extension-for-pytorch dependencies are in oneapi package, before install intel-extension-for-pytorch, we should install oneapi first. Please refer to [Installation Guide](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu&version=v2.1.10%2Bxpu) to install the OneAPI to "/opt/intel folder".

### Create Environment​
Pytorch and Intel-extension-for-pytorch version for intel GPU > 2.1 are required, python version requests equal or higher than 3.9 due to [text evaluation library](https://github.com/EleutherAI/lm-evaluation-harness/tree/master) limitation, the dependent packages are listed in requirements_GPU.txt, we recommend create environment as the following steps. For Intel-exension-for-pytorch, we should install from source code now, and Intel-extension-for-pytorch will add weight-only quantization in the next version.

>**Note**: please install transformers==4.35.2.

```bash
pip install -r requirements_GPU.txt
pip install transformers==4.35.2
source /opt/intel/oneapi/setvars.sh
git clone https://github.com/intel/intel-extension-for-pytorch.git ipex-gpu
cd ipex-gpu
git checkout -b dev/QLLM origin/dev/QLLM
git submodule update --init --recursive
export USE_AOT_DEVLIST='pvc,ats-m150'
export BUILD_WITH_CPU=OFF

python setup.py install
```

## Run
The followings are command to show how to use it.

### 1. Performance
``` bash
# fp16
python run_generation_gpu_woq.py \
    --model EleutherAI/gpt-j-6b \
    --benchmark

# weightonlyquant
python run_generation_gpu_woq.py \
    --model EleutherAI/gpt-j-6b \
    --woq \
    --benchmark
```
> Note: If your device memory is not enough, please quantize and save the model first, then rerun the example with loading the model as below, If your device memory is enough, skip below instruction, just quantization and inference.
```bash
# First step: Quantize and save model
python run_generation_gpu_woq.py \
    --model EleutherAI/gpt-j-6b \
    --woq \ # default quantize method is Rtn
    --output_dir "saved_dir"

# Second step: Load model and inference
python run_generation_gpu_woq.py \
    --model "saved_dir" \
    --benchmark
```

### 2. Accuracy
```bash
# fp16
# quantized model by following the steps above
python run_generation_gpu_woq.py \
    --model "saved_dir" \
    --accuracy \
    --tasks "lambada_openai"
```
