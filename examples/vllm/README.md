# vLLM Acceleration

## Installation
`pip install -r requirement.txt`

## Usage Example
```python
numactl -m 0 -C 0-55 python vllm_acceleration_example.py --model_path=/home/model/chatglm2-6b --prompt=你好
```
