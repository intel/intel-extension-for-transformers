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
cd /itrex/workflows/chatbot/demo/frontend/
npm install
npm run dev &
```

## Run Chat Backend

```
cd /itrex/workflows/chatbot/demo/backend/chat/
```

### Modify run_llama7b.sh/run_gptj6b.sh
Modify the model path in run scripts.

### Run the char server

Run LLaMa-7B model:
```
nohup bash run_llama7b.sh & 
```

Run GPT-J-6B model：
```
nohup bash run_gptj6b.sh &
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
cd /itrex/workflows/chatbot/demo/backend/fastrag/
```

FastRAG backend uses the official llama-7b model or the optimized model.
Modify the model path in run scripts.

### Run the sd server
nohup bash run.sh &

## Run sd Backend

```
cd /itrex/workflows/chatbot/demo/backend/sd/
```

### Run the sd server
nohup bash run.sh &

