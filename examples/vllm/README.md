# vLLM Acceleration with ITREX

This guide explains how to leverage Intel extension for transformers(ITREX) for transformers to accelerate vLLM inference on CPUs. ITREX integrates the vLLM CPU backend and offers optional [QBits acceleration](../../docs/qbits.md) for supported models.

## Installation Methods

1. vLLM Installation with CPU: Install vLLM from source code following the instructions provided [here](https://docs.vllm.ai/en/latest/getting_started/cpu-installation.html).

2. ITREX and Dependencies: Install the ITREX and its dependencies in the local directory.

Note: torch==2.3.0+cpu is required and vllm==0.4.2+cpu is validated.

## Usage Example

ITREX provides a script (vllm_acceleration_example.py) demonstrating accelerated vLLM inference. Run it with the following command:
```bash
numactl -m 0 -C 0-55 python vllm_acceleration_example.py --model_path=/home/model/chatglm2-6b --prompt=你好
```

## Supported and Validated Models
All models listed in the [vLLM Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html) can be accelerated theoretically.

We have validated the majority of existing models using vLLM==0.4.2+cpu:
* [THUDM/chatglm2-6b](https://hf-mirror.com/THUDM/chatglm2-6b)
* [meta-llama/Llama-2-7b-chat-hf](https://hf-mirror.com/meta-llama/Llama-2-7b-chat-hf)
* [baichuan-inc/Baichuan2-7B-Chat](https://hf-mirror.com/baichuan-inc/Baichuan2-7B-Chat)

If you encounter any problems, please let us know.
