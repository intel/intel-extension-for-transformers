# vLLM Acceleration

Intel extension for transformers(ITREX) integrates the vLLM CPU backend, and provides the [qbits acceleration](../../docs/qbits.md) for most vLLM supported models.

## Installation


1. Install vLLM from the scource code, and check this link [installation with CPU](https://docs.vllm.ai/en/latest/getting_started/cpu-installation.html).

2. Install the ITREX and some current directory required packages `pip install -r requirement.txt`.

Note: torch==2.3.0+cpu is required.

## Usage Example
ITREX provides a script which shows the example to accelerate the vLLM inference.
```python
numactl -m 0 -C 0-55 python vllm_acceleration_example.py --model_path=/home/model/chatglm2-6b --prompt=你好
```
