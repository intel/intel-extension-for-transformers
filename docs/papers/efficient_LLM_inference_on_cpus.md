# Efficient LLM Inference on CPUs

In this tutorial, we will demonstrate how to reproduce data in paper [Efficient LLM Inference on CPUs](https://arxiv.org/pdf/2311.00502.pdf). 


## System Summary

Test by Intel on 09/19/2023. 1-node, 1x Intel(R) Xeon(R) Platinum 8480+ @3.8GHz, 56 cores/socket, HT On, Turbo On, Total Memory 256GB (16x16GB DDR5 4800 MT/s [4800 MT/s]), BIOS 3A14.TEL2P1, microcode 0x2b0001b0, CentOS Stream 8, gcc (GCC) 8.5.0 20210514 (Red Hat 8.5.0-10), DL Models, Frameworks/Backends: PyTorch/ONNXRT/LLM Runtime/GGML, Datatype: FP32/INT8/BF16/FP8. Using 1 socket, 56 cores/instance, 1 instance and batch size 1.

Performance varies by use, configuration and other factors. For more complete information about performance and benchmark results, visit www.intel.com/benchmarks


## Run Performance Step by Step

### Prepare Intel Extension for Transformers

Build from source

```shell
git clone https://github.com/intel/intel-extension-for-transformers.git
cd intel-extension-for-transformers/intel_extension_for_transformers/llm/runtime/graph
git submodule update --init --recursive
mkdir build
cd build
cmake .. -G Ninja
ninja
```

### FP32 Inference (Baseline)

>**Note**: Please donwload the corresponding AI model from [huggingface hub](https://huggingface.co/models) before executing following command.


#### 1. Convert Model

Convert Hugginface model. 

Please make sure you have donwloaded the model into local path

```shell
cd intel-extension-for-transformers/intel_extension_for_transformers/llm/runtime/graph
pip install -r requirements.txt
python scripts/convert.py local_model_path --outtype f32 --outfile model_f32.bin
```

#### 2. Inference

Please replace `build/bin/run_<model_name>` with your model name, and fill in `-p <prompt>` with your prompt. We provide several [prompts](../../intel_extension_for_transformers/llm/runtime/graph/scripts/ci/cpp_graph_prompts.json) for different input length. For more details about paramters and their meanings, please go to [argument description](../../intel_extension_for_transformers/llm/runtime/graph/README.md#2-inference-llm)

When running inference, we recommend using `numactl` to control CPU cores in instance. In this paper, we use 56 cores/socket Intel(R) Xeon(R) Platinum 8480+ server, and we recommend setting `cores-per-instance=48` (best performance from our practice). And you can try to find out the best settings on your server.

```shell
OMP_NUM_THREADS=48 numactl -m 0 -C 0-<cores-per-instance> ./build/bin/run_<model_name> -m model_f32.bin  -p <prompt> -n 32 -t 48
```

### INT4 Inference

>**Note**: Please donwload the corresponding AI model from [huggingface hub](https://huggingface.co/models) before executing following command. For converting models, please see above [Convert Model](#1-convert-model)

#### 1. Quantization

Quantize the converted FP32 model with INT4 as weight datatype, INT8 as compute datatype and 128 as group size.

Please select `group_size` between 32 or 128. For more details about parameters and their meanings, please go to [argument description](../../intel_extension_for_transformers/llm/runtime/graph/README.md#1-convert-and-quantize-llm)

```shell
./build/bin/quant_llama  --model_file model_f32.bin --out_file model_q4j128.bin --weight_dtype int4 --group_size 128 --compute_dtype int8 --nthread 24
```

#### 2. Inference

Please replace `build/bin/run_<model_name>` with your model name, and fill in `-p <prompt>` with your prompt. We provide several [prompts](../../intel_extension_for_transformers/llm/runtime/graph/scripts/ci/cpp_graph_prompts.json) for different input length. For more details about paramters and their meanings, please go to [argument description](../../intel_extension_for_transformers/llm/runtime/graph/README.md#2-inference-llm)

When running inference, we recommend using `numactl` to control CPU cores in instance. In this paper, we use 56 cores/socket Intel(R) Xeon(R) Platinum 8480+ server, and we recommend setting `cores-per-instance=48` (best performance from our practice). And you can try to find out the best settings on your server.

```shell
OMP_NUM_THREADS=48 numactl -m 0 -C 0-47 ./build/bin/run_<model_name> -m model_q4j128.bin  -p <prompt> -n 32 -t 48
```


## Run Accuracy Step by Step

### Prepare Environment

```shell
# Install Intel Extension for Transformers
pip install intel-extension-for-transformers
# Install requirements for running accuracy
git clone https://github.com/intel/intel-extension-for-transformers.git
cd examples/huggingface/pytorch/text-generation/quantization
pip install -r requirements.txt
```

>**Note**: If `ImportError: /lib64/libstdc++.so.6: version ``GLIBCXX_3.4.29`` not found` error raised when import intel-extension-for-pytorch, it is due to the high gcc library request, there is the solution to find the correct version.
> ```bash
> find $CONDA_PREFIX | grep libstdc++.so.6
> export LD_PRELOAD=<the path of libstdc++.so.6>:${LD_PRELOAD}
> ```

### FP32 Accuracy (Baseline)

There are four tasks/datasets can be selected to run accuracy: "lambada_openai", "piqa", "helloswag" and "winogrande", and you can choose one or more to get the final results. Here we take [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) as an example.

```shell
python run_generation.py \
           --model meta-llama/Llama-2-7b-hf \
           --accuracy \
           --batch_size 56 \
           --tasks "lambada_openai", "piqa", "hellaswag", "winogrande"
```

### INT4 Accuracy

Quantize the model with INT4 as weight datatype and group size is 128. 

Please select `woq_group_size` between 32 or 128, and set `woq_weight_dtype` to `int4_clip`

```shell
python run_generation.py \
           --model meta-llama/Llama-2-7b-hf \
           --output_dir  saved_results \
           --woq \
           --woq_weight_dtype int4_clip \
           --woq_group_size 128 \
           --accuracy \
           --batch_size 56 \
           --tasks "lambada_openai", "piqa", "hellaswag", "winogrande"
```

