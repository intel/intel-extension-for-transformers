# vllm serving for NeuralChat

vllm is a high-throughput and memory-efficient inference and serving engine for LLMs. It provides efficient management of attention key and value memory with PagedAttention and continuous batching of incoming requests. Here we will show you how to deploy a NeuralChat textbot with vllm.

We currently keep the native GPU-only implementation from [vllm](https://github.com/vllm-project/vllm) and may support CPU based optimization in the future.


## Prepare the environment

```
pip install vllm
```

## Deploy a textbot with vllm

```
neuralchat_server start --config_file textbot_vllm.yaml
```

## Get the result

```
curl -X POST -H "Content-Type: application/json" -d '{"prompt": "Tell me about Intel Xeon processors."}' http://localhost:8000/v1/chat/completions
```
