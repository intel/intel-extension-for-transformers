# Setup Environment with Docker
Chatbot inference dockerfile is based on Ubuntu 22.04.

## Xeon
Option 1 (default): you could use docker build to build the docker image in your environment.
```
docker build --network=host --tag chatbotinfer:latest  ./ -f ./intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/docker/Dockerfile  --target cpu
```

Option 2: If you need to use proxy, please use the following command.
```
docker build --network=host --tag chatbotinfer:latest  ./ --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${http_proxy} -f ./intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/docker/Dockerfile  --target cpu  
```

Then run docker as follows.
```
docker run -it chatbotinfer:latest
```

Note: If you need to use proxy, add `-e http_proxy=${http_proxy} -e https_proxy=${https_proxy}`


## Habana Gaudi
Build the docker image:
```
DOCKER_BUILDKIT=1 docker build --network=host --tag chatbothabana:latest  ./ -f ./intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/docker/Dockerfile  --target hpu --build-arg BASE_NAME="base-installer-ubuntu22.04" --build-arg ARTIFACTORY_URL="vault.habana.ai" --build-arg VERSION="1.11.0" --build-arg REVISION="587" --build-arg PT_VERSION="2.0.1" --build-arg OS_NUMBER="2204"
```

Launch the docker image:
```
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host chatbothabana:latest 
```
