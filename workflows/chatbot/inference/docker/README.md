Chatbot Inference Dockerfile on Ubuntu 22.04

# Setup Environment with Docker

## Xeon
Option 1 (default): you could use docker build to build the docker image in your environment.
```
docker build --network=host --tag chatbotinfer:latest  ./ -f Dockerfile  --target cpu
```

Option 2: If you need to use proxy, please use the following command.
```
docker build --network=host --tag chatbotinfer:latest  ./ --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${http_proxy} -f Dockerfile  --target cpu  
```

```
docker run -it -v ${host_dir}:${mount_dir} -v /dev/shm/models--google--flan-t5-xl:/root/.cache/models--google--flan-t5-xl chatbotinfer:latest
```

If you have already cached the original model and the lora model, you may replace the `-v` parameter to map the cached models on your host machine to the location inside your Docker container.

Note: `${host_dir}` is your local directory, `${mount_dir}` is the docker's directory. If you need to use proxy, add `-e http_proxy=${http_proxy} -e https_proxy=${https_proxy}`


## Habana Gaudi
```
DOCKER_BUILDKIT=1 docker build --network=host --tag chatbothabana:latest  ./ -f Dockerfile  --target hpu --build-arg BASE_NAME="base-installer-ubuntu22.04" --build-arg ARTIFACTORY_URL="vault.habana.ai" --build-arg VERSION="1.11.0" --build-arg REVISION="587" --build-arg PT_VERSION="2.0.1" --build-arg OS_NUMBER="2204"
```

```
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host chatbothabana:latest 
```
