Intel Chatbot Inference Dockerfile installer for Ubuntu22.04

# Do chatbot inference with Docker

## Build

```
docker build ./ --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${http_proxy} -f Dockerfile -t chatbotinfer:latest
```

## Run

### 1. Run the container, establish the mapping of the model files and enter into the container

```
docker run -it -v /dev/shm/models--google--flan-t5-xl:/root/.cache/models--google--flan-t5-xl -v .:/root/chatbot chatbotinfer:latest
```

If you have already cached the original model and the lora model, you may replace the `-v` parameter to map the cached models on your host machine to the location inside your Docker container.

### 2. Inside the container, do the inference

```
python generate.py \
        --base_model_path "google/flan-t5-xl" \
        --lora_model_path "./flan-t5-xl_peft_finetuned_model" \
        --instructions "Transform the following sentence into one that shows contrast. The tree is rotten."
```
