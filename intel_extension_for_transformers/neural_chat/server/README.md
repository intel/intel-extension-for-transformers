# NeuralChat Server Command Line

The simplest approach to use NeuralChat Server including server and client.

## NeuralChat Server
### Help
```bash
neuralchat_server help
```
### Start the server
- Command Line (Recommended)

NeuralChat provides a default chatbot configuration in `./config/neuralchat.yaml`. User could customize the behavior of this chatbot by modifying the value of these fields in the configuration file to specify which LLM model and plugins to be used.

| Fields                    | Sub Fields               | Default Values                             | Possible Values                  |
| ------------------------- | ------------------------ | --------------------------------------- | --------------------------------- |
| host                      |                          | 0.0.0.0                                 | Any valid IP address              |
| port                      |                          | 8000                                    | Any valid port number             |
| model_name_or_path        |                          | "Intel/neural-chat-7b-v3-1"         | A valid model name or path        |
| tokenizer_name_or_path    |                          | ""                                      | A tokenizer name or path          |
| peft_model_path           |                          | ""                                      | A peft model path                 |
| device                    |                          | "auto"                                  | "cpu", "hpu", "xpu", "cuda"       |
| asr                       | enable                   | false                                   | true, false                       |
|                           | args.device              | "cpu"                                   | "cpu", "hpu", "xpu", "cuda"       |
|                           | args.model_name_or_path  | "openai/whisper-small"                  | A valid ASR model name or path    |
|                           | args.bf16                | false                                   | true, false                       |
| tts                       | enable                   | false                                   | true, false                       |
|                           | args.device              | "cpu"                                   | "cpu", "hpu", "xpu", "cuda"       |
|                           | args.voice               | "default"                               | A valid TTS voice                 |
|                           | args.stream_mode         | false                                   | true, false                       |
|                           | args.output_audio_path   | "./output_audio.wav"                    | A valid file path                 |
| tts_multilang             | enable                   | false                                   | true, false                       |
|                           | args.device              | "cpu"                                   | "cpu", "hpu", "xpu", "cuda"       |
|                           | args.spk_id              | 0                                       | A valid speaker ID                |
|                           | args.stream_mode         | false                                   | true, false                       |
|                           | args.output_audio_path   | "./output_audio.wav"                    | A valid file path                 |
| retrieval                 | enable                   | false                                   | true, false                       |
|                           | args.retrieval_type      | "dense"                                 | "dense", other retrieval types   |
|                           | args.input_path          | "../../assets/docs/"                   | A valid input path                |
|                           | args.embedding_model     | "hkunlp/instructor-large"              | A valid embedding model           |
|                           | args.persist_dir         | "./output"                              | A valid directory path            |
|                           | args.max_length          | 512                                     | Any valid integer                 |
|                           | args.process             | false                                   | true, false                       |
| cache                     | enable                   | false                                   | true, false                       |
|                           | args.config_dir          | "../../pipeline/plugins/caching/cache_config.yaml" | A valid directory path |
|                           | args.embedding_model_dir | "hkunlp/instructor-large"              | A valid directory path             |
| safety_checker            | enable                   | false                                   | true, false                       |
| ner                       | enable                   | false                                   | true, false                       |
|                           | args.model_path          | "Intel/neural-chat-7b-v3-1"        | A valid directory path of llm model   |
|                           | args.spacy_model         | "en_core_web_lg"                       | A valid name of downloaded spacy model      |
|                           | args.bf16                | false                                   | true, false                          |
| ner_int                   | enable                   | false                                   | true, false                          |
|                           | args.model_path          | "Intel/neural-chat-7b-v3-1"        | A valid directory path of llm model      |
|                           | args.spacy_model         | "en_core_web_lg"                       | A valid name of downloaded spacy model   |
|                           | args.compute_dtype       | "fp32"                                  | "fp32", "int8"                       |
|                           | args.weight_dtype        | "int8"                                  | "int8", "int4"                       |
| tasks_list                |                          | ['textchat', 'retrieval']              | List of task names, including 'textchat', 'voicechat', 'retrieval', 'text2image', 'finetune', 'photoai'                |



First set the service-related configuration parameters, similar to `./config/neuralchat.yaml`. Set `tasks_list`, which represents the supported tasks included in the service to be started.
**Note:** If the service can be started normally in the container, but the client access IP is unreachable, you can try to replace the `host` address in the configuration file with the local IP address.

Then start the service:
```bash
neuralchat_server start --config_file ./server/config/neuralchat.yaml
```

- Python API
```python
from neuralchat.server.neuralchat_server import NeuralChatServerExecutor

server_executor = NeuralChatServerExecutor()
server_executor(
    config_file="./config/neuralchat.yaml", 
    log_file="./log/neuralchat.log")
```

## NeuralChat Client

### Help
```bash
neuralchat_client help
```
### Access text chat service

- Command Line
```bash
neuralchat_client textchat --server_ip 127.0.0.1 --port 8000 --prompt "Tell me about Intel Xeon processors."
```

- Python API
```python
from neuralchat.server.neuralchat_client import TextChatClientExecutor
import json

executor = TextChatClientExecutor()
executor(
    prompt="Tell me about Intel Xeon processors.",
    server_ip="127.0.0.1",
    port=8000)
```

### Access voice chat service
```bash
neuralchat_client voicechat --server_ip 127.0.0.1 --port 8000 --input say_hello.wav --output response.wav
```

### Access retrieval service
```bash
neuralchat_client retrieval --server_ip 127.0.0.1 --port 8000 --input ./docs/
```
