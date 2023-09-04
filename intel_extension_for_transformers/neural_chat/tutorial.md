Tutorial
==============

This tutorial is used to introduce how to create a chatbot with NeuralChat in few minutes on different devices.

## Basic Usage

NeuralChat provides easy-of-use APIs for user to quickly create a chatbot on local mode or server mode.

```python
## create a chatbot on local mode
from neural_chat import build_chatbot, PipelineConfig

config = PipelineConfig()
chatbot = build_chatbot(config)

## use chatbot to do prediction
response = chatbot.predict("Tell me about Intel Xeon Scalable Processors.")
```

```shell
## create a chatbot on server mode
neuralchat_server start --config_file ./server/config/neuralchat.yaml

```

NeuralChat provides a default chatbot configuration in `neuralchat.yaml`. User could customize the behavior of this chatbot by modifying the value of these fields in the configuration file to specify which LLM model and plugins to be used.

| Fields                    | Sub Fields               | Values                                  |
| ------------------------- | ------------------------ | --------------------------------------- |
| host                      |                          | 0.0.0.0                                 |
| port                      |                          | 8000                                    |
| audio                     | audio_output             | TRUE                                    |
|                           | language                 | "english"                               |
| retrieval                 | retrieval_type           | "dense"                                 |
|                           | retrieval_document_path  | ../../assets/docs/                      |
| caching                   | cache_chat_config_file   | ../../plugins/caching/cache_config.yaml |
|                           | cache_embedding_model_dir| hkunlp/instructor-large                 |
| model_name                |                          | meta-llama/Llama-2-7b-chat-hf           |

## Deply on Different Platforms

### Intel XEON Scalable Processors

On Intel XEON platforms, especially those having [Intel(R) AMX](https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/advanced-matrix-extensions/overview.html) support, user can utilize `mixed precision` feature to accelerate the inference of NeuralChat on local mode, as shown in the below code.

```python
## create a chatbot on local mode
from neural_chat import build_chatbot, PipelineConfig, AMPConfig
import torch

config = PipelineConfig(
            optimization_config=AMPConfig(torch.bfloat16)
            )

chatbot = build_chatbot(config)

## use chatbot to do prediction
response = chatbot.predict("Tell me about Intel Xeon Scalable Processors.")
```

This way converts the underneath LLM model into `bfloat16` on the fly and fully leverage `Intel(R) AMX` technology's power to speed up the inference of the whole working flow.

### Nvidia DataCenter GPU

To leverage Nvidia DataCenter GPU for NeuralChat inference on local mode, user can use below code.

```python
## create a chatbot on local mode
from neural_chat import build_chatbot, PipelineConfig, AMPConfig
import torch

config = PipelineConfig(
            device='cuda:0',
            optimization_config=AMPConfig(torch.float16)
            )


chatbot = build_chatbot(config)

## use chatbot to do prediction
response = chatbot.predict("Tell me about Intel Xeon Scalable Processors.")
```

### Nvidia Client GPU

For Nvidia Client GPU with limited memory and compute power, user can utilize BitsAndBytes to quantize the NeuralChat and then run inference on local mode, as shown in the below code.

```python
## create a chatbot on local mode
from neural_chat import build_chatbot, PipelineConfig
from transformers import BitsAndBytesConfig
import torch

config = PipelineConfig(
            device='cuda:0',
            optimization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16  # or torch.bfloat16
        )
    )

chatbot = build_chatbot(config)

## use chatbot to do prediction
response = chatbot.predict("Tell me about Intel Xeon Scalable Processors.")
```

