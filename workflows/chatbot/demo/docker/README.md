Intel Chatbot Demo Dockerfile installer for Ubuntu22.04

# Build Docker Image

Option 1 (default): you could use docker build to build the docker image in your environment.
```
docker build ./ -f Dockerfile -t chatbotdemo:latest
```

Option 2: If you need to use proxy, please use the following command.
```
docker build ./ --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${http_proxy} -f Dockerfile -t chatbotdemo:latest
```

# Run the Demo

## Get the LLM model
You have the option to download either the official llama-7b, mpt-7b or gpt-j-6b model for chatbot backend. Additionally, you can contact us via email(inc.maintainers@intel.com) to acquire the optimized models.

```
mkdir models
cd models
git clone https://huggingface.co/decapoda-research/llama-7b-hf
git clone https://huggingface.co/EleutherAI/gpt-j-6b
git clone https://huggingface.co/mosaicml/mpt-7b
```


## Run the Docker Image

```
docker run --privileged -v `pwd`/models:/models -it chatbotdemo:latest
```

## Run Frontend

```
cd /itrex/workflows/chatbot/demo/advanced_frontend/
npm install
nohup npm run dev &
```

## Run Chat Backend

```
cd /itrex/workflows/chatbot/inference/backend/chat/
git-lfs install
git clone https://huggingface.co/hkunlp/instructor-large
```

### Modify run_ipex.sh
Modify the `--model-path`  in run scripts. If needed, change the port of controller and model worker.

### Run the char server

Run LLaMa-7B/GPT-J-6B model:
```
nohup bash run_ipex.sh & 
```


## Run FastRAG Backend

### Run elasticsearch

```
su fastrag  # run elasticsearch with non-root user
cd /home/fastrag/elasticsearch-7.17.10
./bin/elasticsearch &
```

### Modify run.sh

```
su -
cd /itrex/workflows/chatbot/inference/backend/fastrag/
```

FastRAG backend uses the official mpt-7b-chat model or the optimized model.
Modify the model path in scripts `run.sh`.

### Run the sd server
```
nohup bash run.sh &
```

## Run sd Backend

```
cd /itrex/workflows/chatbot/inference/backend/sd/
```

### Run the sd server
```
nohup bash run.sh &
```
