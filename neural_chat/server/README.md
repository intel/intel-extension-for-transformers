# NeuralChat Server Command Line

The simplest approach to use NeuralChat Server including server and client.

## NeuralChat Server
### Help
```bash
neuralchat_server help
```
### Start the server
- Command Line (Recommended)

First set the service-related configuration parameters, similar to `./conf/neuralchat.yaml`. Set `tasks_list`, which represents the supported tasks included in the service to be started.
**Note:** If the service can be started normally in the container, but the client access IP is unreachable, you can try to replace the `host` address in the configuration file with the local IP address.

Then start the service:
```bash
neuralchat_server start --config_file ./server/conf/neuralchat.yaml
```

- Python API
```python
from neuralchat.server.neuralchat_server import NeuralChatServerExecutor

server_executor = NeuralChatServerExecutor()
server_executor(
    config_file="./conf/neuralchat.yaml", 
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
