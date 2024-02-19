# Docker
Follow these instructions to set up and run our provided Docker image.

## Set Up Docker Image
Build or Pull the provided docker images.

### Prepare Intel Extension for Transformers
```bash
git clone https://github.com/intel/intel-extension-for-transformers.git itrex
cd itrex
```

### Build Docker Image on CPU
```bash
cd itrex
docker build -f docker/Dockerfile_chatbot --target cpu --network=host -t chatbot:latest .
```
If you need to use proxy, please use the following command
```bash
docker build --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${http_proxy} --target cpu -f docker/Dockerfile_chatbot -t chatbot:latest .
```

### Build Docker Image on HPU
```bash
cd itrex
docker build -f docker/Dockerfile_chatbot --target hpu -t chatbot:latest .
```

### Pull From Docker Hub
```bash
docker pull intel/ai-tools:itrex-chatbot
```


## Use Docker Image
Utilize the docker container based on docker image.
```bash
docker run -itd --net=host --ipc=host intel/ai-tools:itrex-chatbot /bin/bash
docker exec -it <container_id> /bin/bash
```

## Run Simple Test
```bash
docker exec -it <container_id> /bin/bash
export no_proxy=localhost,127.0.0.1,intel.com
cd intel_extension_for_transformers/neural_chat/tests
python server/test_textchat_server.py
```
