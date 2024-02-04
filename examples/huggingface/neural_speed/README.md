# Step-by-Step

To get better performance of popular large language models (LLM), we recommend using [Neural Speed](https://github.com/intel/neural-speed.git), an innovated library designed to provide the most efficient inference of LLMs. Here, we provide the scripts `run_inference.py` for inference, and `run_accuracy.py` for accuracy evaluation. 


# Prerequisite​

We recommend install [Neural Speed](https://github.com/intel/neural-speed.git) from source code to fully leverage the latest features. [How-to-install-neural-speed](https://github.com/intel/neural-speed?tab=readme-ov-file#build-python-package-recommended-way)

> Note: To build neural-speed from source code, GCC higher than 10 is required. If you can't upgrade system GCC, here is a solution using conda install.
> ```bash
> compiler_version==13.1
> conda install --update-deps -c conda-forge gxx==${compiler_version} gcc==${compiler_version} gxx_linux-64==${compiler_version} libstdcxx-ng sysroot_linux-64 -y
> ```

To running accuracy evaluation, python >=3.9, < 3.11 is required due to [text evaluation library](https://github.com/EleutherAI/lm-evaluation-harness/tree/master) limitation.


Other third-party dependencies and versions are listed in requirements, please follow the steps below:


```bash
pip install neural-speed==0.2.dev0
pip install intel-extension-for-transformers==1.3.1
pip install -r requirements.txt
```

# Run


> Note: Please prepare LLMs and save locally before running inference. Here are the models that are currently supported [Support models](https://github.com/intel/neural-speed/blob/main/docs/supported_models.md), you can replace Llama2 in the example with the model in the link.


## 1. Performance

### INT4 Inference
``` bash
# int4 with group-size=32
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_inference.py \
    --model_path ./Llama2-fp32 \
    --prompt "Once upon a time, there existed a little girl," \
    --max_new_tokens 32 \
    --group_size 128
```

### FP32 Inference

``` bash
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_inference.py \
    --model_path ./Llama2-fp32 \
    --prompt "Once upon a time, there existed a little girl," \
    --max_new_tokens 32 \
    --not_quant
```


## 2. Accuracy


### INT4 Accuracy

```bash
# int4 with group-size=32
python run_accuracy.py \
    --model_name ./Llama2-fp32 \
    --tasks "lambada_openai" \
    --use_gptq
```

### FP32 Accuracy

```bash
python run_accuracy.py \
    --model_name ./Llama2-fp32 \
    --tasks "lambada_openai" \
    --model_format "torch"
```
