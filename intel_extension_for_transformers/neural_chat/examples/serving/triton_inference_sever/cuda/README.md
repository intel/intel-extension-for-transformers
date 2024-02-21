# Serving NeuralChat Text Generation with Triton Inference Server (CUDA)

Nvidia Triton Inference Server is a widely adopted inference serving software. We also support serving and deploying NeuralChat models with Triton Inference Server on CUDA devices.

## Prepare serving scripts

```
cd <path to intel_extension_for_transformers>/neural_chat/examples/serving
mkdir -p models/text_generation/1/
cp ../../serving/triton/text_generation/cuda/model.py models/text_generation/1/model.py
cp ../../serving/triton/text_generation/cuda/config.pbtxt models/text_generation/config.pbtxt
```


Then your folder structure under the current `serving` folder should be like:

```
serving/
├── models
│   └── text_generation
│       ├── 1
│       │   ├── model.py
│       └── config.pbtxt
├── README.md
```

## Start Triton Inference Server

```
cd <path to intel_extension_for_transformers>/neural_chat/examples/serving
docker run -d --gpus all -e PYTHONPATH=/opt/tritonserver/intel-extension-for-transformers --net=host -v ${PWD}/models:/models spycsh/triton_neuralchat_gpu:v2 tritonserver --model-repository=/models --http-port 8021
```

Pass `-v` to map your model on your host machine to the docker container.

## Multi-card serving (optional)

You can also do multi-card serving to get better throughput by specifying a instance group provided by Triton Inference Server.

To do that, please edit the the field `instance_group` in your `config.pbtxt`.

One example would be like following:

```
instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [0, 1]
  }
]
```

This means for every gpu device, we initialize an execution instance. Please check configuration details through this [link](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#multiple-model-instances).

## Quick check whether the server is up

To check whether the server is up:

```
curl -v localhost:8021/v2/health/ready
```

You will find a `HTTP/1.1 200 OK` if your server is up and ready for receiving requests.

## Use Triton client to send inference request

Start the Triton client and enter into the container

```
cd <path to intel_extension_for_transformers>/neural_chat/examples/serving
docker run --gpus all --net=host -it --rm -v ${PWD}/../../serving/triton/text_generation/client.py:/workspace/text_generation/client.py nvcr.io/nvidia/tritonserver:23.11-py3-sdk
```

Send a request

```
python /workspace/text_generation/client.py --prompt="Tell me about Intel Xeon Scalable Processors." --url=localhost:8021
```
