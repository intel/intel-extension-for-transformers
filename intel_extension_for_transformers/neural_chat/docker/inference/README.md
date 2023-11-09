Intel Neural Chat Inference Dockerfile installer for Ubuntu22.04

# Do chatbot inference with Docker

## Environment Setup

### Setup Xeon SPR Environment

```bash
cd /path/to/workspace/intel-extension-for-transformers
docker build  --no-cache ./ --target cpu \ 
        --build-arg REPO="${you_repo_path}" \  # Optional, set repository(default: https://github.com/intel/intel-extension-for-transformers.git)
        --build-arg ITREX_VER="${branch}" \  # Optional, set branch(default: main)
        --build-arg http_proxy="${http_proxy}" \  # Optional, use proxy
        --build-arg https_proxy="${https_proxy}" \  # Optional, use proxy
        --build-arg REPO_PATH="." \  # Optional, use local files
        -f /path/to/workspace/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/docker/Dockerfile \ 
        -t chatbotinfer:latest
```

If you need to set proxy settings:

```bash
docker build --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f  /path/to/workspace/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/docker/Dockerfile -t chatbotinfer:latest . --target cpu
```

If you don't need to set proxy settings:

```bash
docker build -f /path/to/workspace/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/docker/Dockerfile -t chatbotinfer:latest . --target cpu
```

If you need to use local files:

```bash
cd /path/to/workspace/intel-extension-for-transformers
docker build -f /path/to/workspace/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/docker/Dockerfile  --build-arg REPO_PATH="." -t chatbotinfer:latest . --target cpu
```

If you need to use forked repository or other branch:

```bash
docker build -f /path/to/workspace/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/docker/Dockerfile --build-arg REPO=<forked_repository> --build-arg ITREX_VER=<your_branch_name> -t chatbotinfer:latest . --target cpu
```

```bash
docker run -it chatbotinfer:latest /bin/bash
```

### Setup Habana Gaudi Environment
```bash
cd /path/to/workspace/intel-extension-for-transformers
docker build  --no-cache ./ --target hpu \ 
        --build-arg REPO="${you_repo_path}" \  # Optional, set repository(default: https://github.com/intel/intel-extension-for-transformers.git)
        --build-arg ITREX_VER="${branch}" \  # Optional, set branch(default: main)
        --build-arg http_proxy="${http_proxy}" \  # Optional, use proxy
        --build-arg https_proxy="${https_proxy}" \  # Optional, use proxy
        --build-arg REPO_PATH="." \  # Optional, use local files
        -f /path/to/workspace/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/docker/Dockerfile \ 
        -t chatbothabana:latest
```

If you need to set proxy settings:

```bash
DOCKER_BUILDKIT=1 docker build --network=host --tag chatbothabana:latest  --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy  ./ -f /path/to/workspace/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/docker/Dockerfile  --target hpu
```

If you don't need to set proxy settings:

```bash
docker build -f /path/to/workspace/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/docker/Dockerfile -t chatbothabana:latest . --target hpu
```

If you need to use local files:

```bash
cd /path/to/workspace/intel-extension-for-transformers
docker build -f /path/to/workspace/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/docker/Dockerfile  --build-arg REPO_PATH="." -t chatbothabana:latest . --target hpu
```

If you need to use forked repository or other branch:

```bash
docker build -f /path/to/workspace/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/docker/Dockerfile --build-arg REPO=<forked_repository> --build-arg ITREX_VER=<your_branch_name> -t chatbothabana:latest . --target hpu
```

```bash
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host chatbothabana:latest /bin/bash
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
