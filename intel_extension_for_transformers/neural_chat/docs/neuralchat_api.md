# Building RESTful API Server
To utilize the RESTful API, the initial step involves launching the RESTful API server, easily achievable by running the rich examples found in the [examples](../examples/deployment/) directory.

NeuralChat provides support for the following RESTful API Servers:

- RESTful API Server for Text Chat
- RESTful API Server for Voice Chat
- RESTful API Server for Retrieval-Augmented Generation
- RESTful API Server for Image and Video Processing
- RESTful API Server for Code Generation


# OpenAI-Compatible RESTful APIs
NeuralChat provides OpenAI-compatible APIs for LLM inference, so you can use NeuralChat as a local drop-in replacement for OpenAI APIs. The NeuralChat server is compatible with both [openai-python library](https://github.com/openai/openai-python) and cURL commands.

The following OpenAI APIs are supported:

- Chat Completions. (Reference: https://platform.openai.com/docs/api-reference/chat)
- Completions. (Reference: https://platform.openai.com/docs/api-reference/completions)


# Additional useful RESTful APIs
In addition to the text-based chat RESTful API, NeuralChat offers several helpful plugins in its RESTful API lineup to aid users in building multimodal applications.
NeuralChat supports the following RESTful APIs:
- Finetuning
- Audio Chat
- Document Retrieval
- Code Generation
- Text to Image
- Image to Image
- Face animation

TODO: Add a RESTful API table here.

# Access the Server using the RESTful API

## OpenAI Official SDK

The RESTful API Server can be used directly with [openai-python library](https://github.com/openai/openai-python).

First, install openai-python:

```bash
pip install --upgrade openai
```

```python
import openai
# to get proper authentication, make sure to use a valid key that's listed in
# the --api-keys flag. if no flag value is provided, the `api_key` will be ignored.
openai.api_key = "EMPTY"
openai.api_base = "http://localhost:80/v1"

model = "Intel/neural-chat-7b-v3-1"
prompt = "Once upon a time"

# create a completion
completion = openai.Completion.create(model=model, prompt=prompt, max_tokens=64)
# print the completion
print(prompt + completion.choices[0].text)

# create a chat completion
completion = openai.ChatCompletion.create(
  model=model,
  messages=[{"role": "user", "content": "Tell me about Intel Xeon Scalable Processors."}]
)
# print the completion
print(completion.choices[0].message.content)
```

## cURL
cURL is another good tool for observing the output of the api.

List Models:

```bash
curl http://localhost:80/v1/models
```

Chat Completions:

```bash
curl http://localhost:80/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Intel/neural-chat-7b-v3-1",
    "messages": [{"role": "user", "content": "Tell me about Intel Xeon Scalable Processors."}]
  }'
```

Text Completions:

```bash
curl http://localhost:80/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Intel/neural-chat-7b-v3-1",
    "prompt": "Once upon a time",
    "max_tokens": 41,
    "temperature": 0.5
  }'
```

