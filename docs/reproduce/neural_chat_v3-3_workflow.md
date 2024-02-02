# Step-by-Step

This is a step-by-step tutorial to obtain the extreme inference speed and accuracy of [Intel/neural-chat-7b-v3-3](https://huggingface.co/Intel/neural-chat-7b-v3), we recommend using [AutoRound](https://github.com/intel/auto-round.git) for quantization and [Neural Speed](https://github.com/intel/neural-speed.git) for inference. 


# Prerequisite​

We recommend install [Neural Speed](https://github.com/intel/neural-speed.git) from source code to fully leverage the latest features.

> Note: To build neural-speed from source code, GCC higher than 10 is required. If you can't upgrade system GCC, here is a solution using conda install.
> ```bash
> compiler_version==13.1
> conda install --update-deps -c conda-forge gxx==${compiler_version} gcc==${compiler_version} gxx_linux-64==${compiler_version} libstdcxx-ng sysroot_linux-64 -y
> ```


```bash
# build neural-speed from source code
git clone https://github.com/intel/neural-speed.git
cd neural-speed
pip install -r requirements.txt
python setup.py install
# come back to current working directory
cd ..
pip install intel-extension-for-transformers==1.3.1
```

# Run
To obtain the extreme inference speed while keep inference accuracy, we will use [AutoRound](../../examples/huggingface/pytorch/text-generation/quantization/auto_round/README.md) algorithm for quantization, and use [Neural Speed](../../examples/huggingface/neural_speed/README.md) for inference.


## 1. Quantization

For quantization using [AutoRound](https://github.com/intel/auto-round.git), we provide [examples and scripts](../../examples/huggingface/pytorch/text-generation/quantization/auto_round/README.md).


```bash
cd examples/huggingface/pytorch/text-generation/quantization/auto_round
pip install -r requirements.txt
bash run_autoround.sh
```

After running the example scripts, the quantized Neural-Chat-V3-3-int4 model will be saved in current working directory with name `neural-chat-v3-3-int4`.


## 2. Inference

For LLM inference using [Neural Speed](https://github.com/intel/neural-speed.git), we provide [examples and scripts](../../examples/huggingface/neural_speed/README.md).


``` bash
cd examples/huggingface/neural_speed
pip install -r requirements.txt
numactl -m <node N> -C <cpu list> python run_inference.py \
    --model_path "neural-chat-v3-3-int4" \
    --prompt "Once upon a time, there existed a little girl," \
    --max_new_tokens 32 \
    --group_size 128 \
    --use_gptq
```

>**Note**: If `ImportError: /lib64/libstdc++.so.6: version ``GLIBCXX_3.4.29`` not found` error raised when import intel-extension-for-pytorch, it is due to the high gcc library request, there is the solution to find the correct version.
> ```bash
> libstdc_path_=$(find $CONDA_PREFIX | grep libstdc++.so.6 | sort | head -1)
> export LD_PRELOAD=${libstdc_path_}:${LD_PRELOAD}
> ```

## 3. Accuracy

We also provide LLM accuracy evaluation [here](../../examples/huggingface/neural_speed/README.md).


To running accuracy evaluation, python >=3.9, < 3.11 is required due to [text evaluation library](https://github.com/EleutherAI/lm-evaluation-harness/tree/master) limitation.


```bash
# still working in examples/huggingface/neural_speed directory
python run_autoround_example.py \
    --model_name "neural-chat-v3-3-int4" \
    --tasks "lambada_openai"
```
