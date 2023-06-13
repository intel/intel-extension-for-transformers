Step-by-Step
============
This document describes the step-by-step instructions to run large language models(LLMs) `float32` and `bfloat16` inference on 4th Gen Intel® Xeon® Scalable Processor (codenamed [Sapphire Rapids](https://www.intel.com/content/www/us/en/products/docs/processors/xeon-accelerated/4th-gen-xeon-scalable-processors.html)). Last word prediction accuracy is provide by [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness.git).


## Prerequisite
### Create Environment
```bash
# Create Environment (conda)
conda create -n llm python=3.9 -y
conda install mkl mkl-include -y
conda install gperftools jemalloc==5.2.1 -c conda-forge -y

# Installation
git clone https://github.com/intel/intel-extension-for-transformers.git itrex
cd itrex
pip install -v .
cd examples/huggingface/pytorch/language-modeling/inference
pip install -r requirements.txt

# Setup Environment Variables
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0
# IOMP
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
# Tcmalloc is a recommended malloc implementation that emphasizes fragmentation avoidance and scalable concurrency support.
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
```

## Run

### Inference

```bash
# "--precision provide two options "bf16"/"fp32"
# "--jit" used to covert model to torchscript mode
# "--ipex" enable intel_extension_for_pytorch
numactl -m <node N> -C <cpu list> \
    python run_clm_no_trainer.py \
        --precision "bf16" \ 
        --model "EleutherAI/gpt-j-6b" \ 
        --accuracy \
        --task "lambada_openai"
```
