# Serving NeuralChat Text Generation with Triton Inference Server

Nvidia Triton Inference Server is a widely adopted inference serving software. We also support serving and deploying NeuralChat models with Triton Inference Server.

## Prepare serving scripts

```
cd <path to intel_extension_for_transformers>/neural_chat/examples/serving
mkdir -p models/text_generation/1/
cp text_generation/model.py models/text_generation/1/model.py
cp text_generation/config.pbtxt models/text_generation/config.pbtxt
```

Then your folder structure under the current `serving` folder shoud be like:

```
serving/
├── models
│   └── text_generation
│       ├── 1
│       │   ├── model.py
│       └── config.pbtxt
├── README.md
└── text_generation
    ├── client.py
    ├── config.pbtxt
    └── model.py
```

## Start Triton Inference Server

```
cd <path to intel_extension_for_transformers>/neural_chat/examples/serving
docker run -d -e PYTHONPATH=/opt/tritonserver/intel-extension-for-transformers --net=host -v ${PWD}/models:/models spycsh/triton_neuralchat:v1 tritonserver --model-repository=/models --http-port 8021
```

NeuralChat by default uses `Intel/neural-chat-7b-v3-1` as the LLM, so if you already have this Huggingface model in cache, you can add a `-v` flag to avoid downloading the model every time you start the server, like follows:

```
docker run -d -e PYTHONPATH=/opt/tritonserver/intel-extension-for-transformers --net=host -v ${PWD}/models:/models -v /root/.cache/huggingface/hub/models--Intel--neural-chat-7b-v3-1:/root/.cache/huggingface/hub/models--Intel--neural-chat-7b-v3-1  spycsh/triton_neuralchat:v1  tritonserver --model-repository=/models --http-port=8021
```

To check whether the server is on:

```
curl -v localhost:8021/v2/health/ready
```

## Use Triton client to send inference request

Start the Triton client and enter into the container

```
cd <path to intel_extension_for_transformers>/neural_chat/examples/serving
docker run --net=host -it --rm -v ${PWD}/text_generation/client.py:/workspace/text_generation/client.py nvcr.io/nvidia/tritonserver:23.11-py3-sdk
```

Send a request

```
python /workspace/text_generation/client.py --prompt="Tell me about Intel Xeon Scalable Processors." --url=localhost:8021
```
