Intel Neural Chat Inference Dockerfile installer for Ubuntu22.04

# Do chatbot inference with Docker

## Environment Setup

### Setup Xeon SPR Environment
Option 1 (default): you could use docker build to build the docker image in your environment. If you need to set proxy settings, add `--build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy` when docker build.
```
docker build --format docker --network=host -t ${IMAGE_NAME}:${IMAGE_TAG}  ./ -f /path/to/workspace/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/docker/Dockerfile  --target cpu
```

Option 2: Download from docker hub.
```
docker pull intel/ai-tools:itrex-chatbot 
```

If you have downloaded models and dataset locally, just mount the files to the docker container using '-v'. Make sure using the absolute path for host_dir.
```
docker run -it -v ${host_dir}:${mount_dir} ${IMAGE_NAME}:${IMAGE_TAG}
```

>**Note**: `${host_dir}` is your local directory, `${mount_dir}` is the docker's directory. If you need to use proxy, add `-e http_proxy=${http_proxy} -e https_proxy=${https_proxy}`


### Setup Habana Gaudi Environment
```
DOCKER_BUILDKIT=1 docker build --format docker --network=host -t ${IMAGE_NAME}:${IMAGE_TAG}  ./ -f Dockerfile  --target hpu --build-arg BASE_NAME="base-installer-ubuntu22.04" --build-arg ARTIFACTORY_URL="vault.habana.ai" --build-arg VERSION="1.11.0" --build-arg REVISION="587" --build-arg PT_VERSION="2.0.1" --build-arg OS_NUMBER="2204"
```

```
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host ${IMAGE_NAME}:${IMAGE_TAG} 
```

## Run the Inference
You can use the generate.py script for performing direct inference on Habana Gaudi instance. We have enabled BF16 to speed up the inference. Please use the following command for inference.

### Run the Inference on Xeon SPR
```
python generate.py \
        --base_model_path "mosaicml/mpt-7b-chat" \
        --use_kv_cache \
        --tokenizer_name "EleutherAI/gpt-neox-20b" \
        --instructions "Transform the following sentence into one that shows contrast. The tree is rotten."
```

Note: You can add the flag `--jit` to use jit trace to accelerate generation.

### Run the Inference on Habana Gaudi
```
python generate.py \
        --base_model_path "mosaicml/mpt-7b-chat" \
        --tokenizer_name "EleutherAI/gpt-neox-20b" \
        --habana \
        --use_hpu_graphs \
        --use_kv_cache \
        --instructions "Transform the following sentence into one that shows contrast. The tree is rotten."
```
