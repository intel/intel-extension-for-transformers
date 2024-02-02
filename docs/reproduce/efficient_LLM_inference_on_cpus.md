# Efficient LLM Inference on CPUs

In this tutorial, we will demonstrate how to reproduce data in NeurIPS'23 paper [Efficient LLM Inference on CPUs](https://arxiv.org/pdf/2311.00502.pdf). 


## System Summary

Test by Intel on 09/19/2023. 1-node, 1x Intel(R) Xeon(R) Platinum 8480+ @3.8GHz, 56 cores/socket, HT On, Turbo On, Total Memory 256GB (16x16GB DDR5 4800 MT/s [4800 MT/s]), BIOS 3A14.TEL2P1, microcode 0x2b0001b0, CentOS Stream 8, gcc (GCC) 8.5.0 20210514 (Red Hat 8.5.0-10), DL Models, Frameworks/Backends: PyTorch/ONNXRT/LLM Runtime/GGML, Datatype: FP32/INT8/BF16/FP8. Using 1 socket, 56 cores/instance, 1 instance and batch size 1.

Performance varies by use, configuration and other factors. For more complete information about performance and benchmark results, visit www.intel.com/benchmarks


## Run Performance Step by Step

### Prepare environment


```shell
pip install intel-extension-for-transformers==1.3.1
pip install neural-speed==0.2
```

### FP32 Inference (Baseline)

>**Note**: Please download the corresponding AI model from [huggingface hub](https://huggingface.co/models) before executing following command.


``` bash
cd examples/huggingface/neural_speed
pip install -r requirements.txt
numactl -m <node N> -C <cpu list> python run_inference_.py \
    --model_path "Model-Path-fp32" \
    --prompt "Once upon a time, there existed a little girl," \
    --max_new_tokens 32 \
    --not_quant
```

### INT4 Inference

>**Note**: Please download the corresponding AI model from [huggingface hub](https://huggingface.co/models) before executing following command.

``` bash
cd examples/huggingface/neural_speed
pip install -r requirements.txt
# int4 with group-size=32
numactl -m <node N> -C <cpu list> python run_inference.py \
    --model_path "Model-Path-int4" \
    --prompt "Once upon a time, there existed a little girl," \
    --max_new_tokens 32 \
    --group_size 32
```

## Run Accuracy Step by Step

### Prepare Environment

```shell
cd examples/huggingface/pytorch/text-generation/quantization
pip install -r requirements.txt
```

>**Note**: If `ImportError: /lib64/libstdc++.so.6: version ``GLIBCXX_3.4.29`` not found` error raised when import intel-extension-for-pytorch, it is due to the high gcc library request, there is the solution to find the correct version.
> ```bash
> find $CONDA_PREFIX | grep libstdc++.so.6
> export LD_PRELOAD=<the path of libstdc++.so.6>:${LD_PRELOAD}
> ```

>**Note**: To running accuracy evaluation, python >=3.9, <= 3.11 is required due to [text evaluation library](https://github.com/EleutherAI/lm-evaluation-harness/tree/master) limitation.



### FP32 Accuracy (Baseline)

There are four tasks/datasets can be selected to run accuracy: "lambada_openai", "piqa", "helloswag" and "winogrande", and you can choose one or more to get the final results. 

```bash
python run_accuracy.py \
    --model_name "Model-Path-fp32" \
    --tasks "lambada_openai" \
    --model_format "torch"
```

### INT4 Accuracy

```bash
# int4 with group-size=32
python run_accuracy.py \
    --model_name ./Llama2 \
    --tasks "lambada_openai"
```
