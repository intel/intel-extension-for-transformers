# Step-by-Step

This is a step-by-step tutorial to obtain the extreme inference speed and accuracy of [Intel/neural-chat-7b-v3-3](https://huggingface.co/Intel/neural-chat-7b-v3-3), we recommend using [AutoRound](https://github.com/intel/auto-round.git) for quantization and [Neural Speed](https://github.com/intel/neural-speed.git) for inference. 


# Prerequisite​
We recommend use the latest version of [Intel-Extension-for-Transformers v1.3.1](https://pypi.org/project/intel-extension-for-transformers/1.3.1/) and [NeualSpeed v0.2.dev0](https://pypi.org/project/neural-speed/0.2.dev0/)

```bash
# install neural-speed
pip install neural-speed==0.2.dev0
# install intel-extension-for-transformers
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

After running the example scripts, the quantized neural-chat-v3-3 model will be saved in `output` directory with the name `neural-chat-v3-3-autoround-int4_GPTQ`.


## 2. Inference

### INT4 inference
For LLM inference using [Neural Speed](https://github.com/intel/neural-speed.git), we provide [examples and scripts](../../examples/huggingface/neural_speed/README.md).


``` bash
cd examples/huggingface/neural_speed
pip install -r requirements.txt
model_path="../pytorch/text-generation/quantization/auto_round/output/neural-chat-7b-v3-3-autoround-int4_GPTQ"
export NEURAL_SPEED_VERBOSE=1
numactl -m <node N> -C <cpu list> python run_inference.py \
    --model_path ${model_path} \
    --prompt "Once upon a time, there existed a little girl," \
    --max_new_tokens 32 \
    --use_gptq
```


### FP32 Inference

>**Note**: Please download the [Intel/neural-chat-7b-v3-3](https://huggingface.co/Intel/neural-chat-7b-v3-3) from huggingface model hub and keep it in local.


``` bash
cd examples/huggingface/neural_speed
pip install -r requirements.txt
# if you keep model in cache, please check if model is restored in "${HOME}/.cache/huggingface/hub/models--Intel--neural-chat-7b-v3-3/snapshots/7b86016aa1d2107440c1928694a7bba926509887" or "${HF_HOME}/.cache/huggingface/hub/models--Intel--neural-chat-7b-v3-3/snapshots/7b86016aa1d2107440c1928694a7bba926509887"
model_path="LOCAL-PATH-TO-NEURAL-CHAT-v3-3"
numactl -m <node N> -C <cpu list> python run_inference.py \
    --model_path "Intel/neural-chat-7b-v3-3" \
    --prompt "Once upon a time, there existed a little girl," \
    --max_new_tokens 32 \
    --not_quant
```



>**Note**: If `ImportError: /lib64/libstdc++.so.6: version ``GLIBCXX_3.4.29`` not found` error raised when import intel-extension-for-pytorch, it is due to the high gcc library request, there is the solution to find the correct version.
> ```bash
> libstdc_path_=$(find $CONDA_PREFIX | grep libstdc++.so.6 | sort | head -1)
> export LD_PRELOAD=${libstdc_path_}:${LD_PRELOAD}
> ```


## 3. Accuracy

We also provide LLM accuracy evaluation [here](../../examples/huggingface/neural_speed/README.md).


To running accuracy evaluation, python >=3.9, < 3.11 is required due to [text evaluation library](https://github.com/EleutherAI/lm-evaluation-harness/tree/master) limitation.


### INT4 Accuracy

```bash
# still working in examples/huggingface/neural_speed directory
model_path="../pytorch/text-generation/quantization/auto_round/output/neural-chat-7b-v3-3-autoround-int4_GPTQ"
python run_accuracy.py \
    --model_name ${model_path} \
    --tasks "lambada_openai" \
    --use_gptq
```


### FP32 Accuracy

```bash
python run_accuracy.py \
    --model_name "Intel/neural-chat-7b-v3-3" \
    --tasks "lambada_openai" \
    --model_format "torch" \
    --batch_size 1
```
