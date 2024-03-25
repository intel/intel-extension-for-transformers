Intel Neural Chat Inference Dockerfile installer for Ubuntu22.04

# Do chatbot inference with Docker

## 1. Docker Image Setup

### Option 1: Build Docker Image

>**Note**: If your docker daemon is too big and cost long time to build docker image, you could create a `.dockerignore` file including useless files to reduce the daemon size.

#### Please clone a ITREX repo to this path.
```bash
git clone https://github.com/intel/intel-extension-for-transformers.git
cd intel-extension-for-transformers
```

If you need to set proxy settings, add `--build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy` when `docker build`.  

If you need to use your branch, add `--build-arg ITREX_VER="${your_branch}` when `docker build`.  

If you need to clone repo in docker (including using forked repository), add `--build-arg REPO="${you_repo_path}"` when `docker build`.  

If you need to use local repository, add `--build-arg REPO_PATH="."` when `docker build`.

#### On Xeon SPR Environment

```bash
docker build --build-arg UBUNTU_VER=22.04 -f intel_extension_for_transformers/neural_chat/docker/Dockerfile -t neuralchat_inference:latest . --target cpu
```

#### On Habana Gaudi Environment

```bash
docker build --build-arg UBUNTU_VER=22.04 -f intel_extension_for_transformers/neural_chat/docker/Dockerfile -t neuralchat_inference:latest . --target hpu
```

#### On Nvidia GPU Environment

```bash
docker build -f intel_extension_for_transformers/neural_chat/docker/Dockerfile -t neuralchat_inference:latest . --target nvgpu
```

### Option 2: Docker Pull from Docker Hub
```bash
docker pull intel/ai-tools:itrex-chatbot
```

## 2. Create Docker Container

### On Xeon SPR Environment

```bash
docker run -it --name="chatbot" neuralchat_inference:latest /bin/bash
```

If you have downloaded models and dataset locally, just mount the files to the docker container using `-v`. Replce `${host_dir}` with your local directory, and `${mount_dir}` with the directory in docker container. Please make sure using the absolute path for `${host_dir}`. 

If you need to set proxy settings, please add `-e https_proxy=$https_proxy -e http_proxy=$http_proxy -e no_proxy="localhost,127.0.0.1"`. 

```bash
docker run -it --name="chatbot" -e https_proxy=$https_proxy -e http_proxy=$http_proxy -e no_proxy="localhost,127.0.0.1" -v ${host_dir}:${mount_dir} neuralchat_inference:latest /bin/bash
```

### On Habana Gaudi Environment

```bash
docker run -it --runtime=habana --name="chatbot" -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none -v /dev/shm:/dev/shm --cap-add=sys_nice --net=host --ipc=host neuralchat_inference:latest /bin/bash
```

If you have downloaded models and dataset locally, just mount the files to the docker container using `-v`. Replce `${host_dir}` with your local directory, and `${mount_dir}` with the directory in docker container. Please make sure using the absolute path for `${host_dir}`. 

If you need to set proxy settings, please add `-e https_proxy=$https_proxy -e http_proxy=$http_proxy -e no_proxy="localhost,127.0.0.1"`. 


```bash
docker run -it --runtime=habana --name="chatbot" -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none -e https_proxy=$https_proxy -e http_proxy=$http_proxy -e no_proxy="localhost,127.0.0.1" -v /dev/shm:/dev/shm  -v ${host_dir}:${mount_dir} --cap-add=sys_nice --net=host --ipc=host neuralchat_inference:latest /bin/bash
```

## 3. Simple Test using Docker Container
```bash
## if you are already inside the container, skip this step
docker exec -it chatbot /bin/bash
## run inference unittest
pip install -r pipeline/plugins/audio/requirements.txt
pip install --upgrade --force-reinstall torch==2.2.0
cd tests/ci/api
python test_inference.py

```


## Run the Inference
You can use the generate.py script for performing direct inference on Habana Gaudi instance. We have enabled BF16 to speed up the inference. Please use the following command for inference.

### Run the Inference on Xeon SPR
```bash
python generate.py \
        --base_model_path "mosaicml/mpt-7b-chat" \
        --use_kv_cache \
        --tokenizer_name "EleutherAI/gpt-neox-20b" \
        --instructions "Transform the following sentence into one that shows contrast. The tree is rotten."
```

Note: You can add the flag `--jit` to use jit trace to accelerate generation.

### Run the Inference on Habana Gaudi
```bash
python generate.py \
        --base_model_path "mosaicml/mpt-7b-chat" \
        --tokenizer_name "EleutherAI/gpt-neox-20b" \
        --habana \
        --use_hpu_graphs \
        --use_kv_cache \
        --instructions "Transform the following sentence into one that shows contrast. The tree is rotten."
```
