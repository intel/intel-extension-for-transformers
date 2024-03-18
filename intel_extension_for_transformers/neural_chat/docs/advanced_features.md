# Plugins

## Chatbot with RAG
NeuralChat introduces 'plugins' that provide a comprehensive range of helpful LLM utilities and features to enhance the chatbot's capabilities. One such plugin is RAG(Retrieval-Augmented Generation), widely utilized in knowledge-based chatbot applications.

Taking inspiration from earlier chatbot frameworks like [langchain](https://github.com/langchain-ai/langchain), [Llama-Index](https://github.com/run-llama/llama_index) and [haystack](https://github.com/deepset-ai/haystack), the NeuralChat API simplifies the creation and utilization of chatbot models, seamlessly integrating the powerful capabilities of RAG. This API design serves as both an easy-to-use extension for langchain users and a user-friendly deployment solution for the general user.

To ensure a seamless user experience, the plugin has been designed to be compatible with common file formats such as txt, xlsx, csv, word, pdf, html and json/jsonl. It's essential to note that for optimal functionality, certain file formats must adhere to specific structural guidelines.

|  File Type   | Predefined Structure  |
|  :----:  | :----:  |
| txt  | NA |
| html  | NA |
| markdown  | NA |
| word  | NA |
| pdf  | NA |
| xlsx  | ['Questions', 'Answers']<br>['question', 'answer', 'link']<br>['context', 'link'] |
| csv  | ['question', 'correct_answer'] |
| json/jsonl  | {'content':xxx, 'link':xxx}|

Consider this straightforward example: by providing the URL of the CES main page, the chatbot can engage in a conversation based on the content from that webpage.

```python
# python code
from intel_extension_for_transformers.neural_chat import build_chatbot, PipelineConfig, plugins
plugins.retrieval.enable = True
plugins.retrieval.args["input_path"]=["https://www.ces.tech/"]
conf = PipelineConfig(plugins=plugins)
chatbot = build_chatbot(conf)
response = chatbot.predict("When is CES 2024?")
print(response)
```

RAG demo video:

https://github.com/intel/intel-extension-for-transformers/assets/104267837/d12c0123-3c89-461b-8456-3b3f03e3f12e

The detailed description about RAG plugin, please refer to [README](./pipeline/plugins/retrieval/README.md)

## Chatbot with Multimodal

NeuralChat integrates multiple plugins to enhance multimodal capabilities in chatbot applications. The Audio Processing and Text-to-Speech (TTS) Plugin is a software component specifically designed to improve audio-related functionalities, especially for TalkingBot. Additionally, NeuralChat supports image and video plugins to facilitate tasks involving image and video generation.

Test audio sample download:

```shell
wget -c https://github.com/intel/intel-extension-for-transformers/blob/main/intel_extension_for_transformers/neural_chat/assets/audio/sample.wav
```

Python Code for Audio Processing and TTS:

```python
# Python code
from intel_extension_for_transformers.neural_chat import build_chatbot, PipelineConfig, plugins
plugins.asr.enable = True
plugins.tts.enable = True
plugins.tts.args["output_audio_path"] = "./response.wav"
pipeline_config = PipelineConfig(plugins=plugins)
chatbot = build_chatbot(pipeline_config)
response = chatbot.predict(query="./sample.wav")
```

Multimodal demo video:

https://github.com/intel/intel-extension-for-transformers/assets/104267837/b5a3f2c4-f7e0-489b-9513-661b400b8983

Please check this [example](./examples/deployment/photo_ai/README.md) for details.

## Code Generation

Code generation represents another significant application of Large Language Model(LLM) technology. NeuralChat supports various popular code generation models across different devices and provides services similar to GitHub Copilot. NeuralChat copilot is a hybrid copilot which involves real-time code generation using client PC combines with deeper server-based insight. Users have the flexibility to deploy a robust Large Language Model (LLM) in the public cloud or on-premises servers, facilitating the generation of extensive code excerpts based on user commands or comments. Additionally, users can employ an optimized LLM on their local PC as an AI assistant capable of addressing queries related to user code, elucidating code segments, refactoring, identifying and rectifying code anomalies, generating unit tests, and more.

Neural Copilot demo video:

https://github.com/intel/intel-extension-for-transformers/assets/104267837/1328969a-e60e-48b9-a1ef-5252279507a7

Please check this [example](./examples/deployment/codegen/README.md) for details.


## Safety Checker

We prioritize the safe and responsible use of NeuralChat for everyone. Nevertheless, owing to the inherent capabilities of large language models (LLMs), we cannot assure that the generated outcomes are consistently safe and beneficial for users. To address this, we've developed a safety checker that meticulously reviews and filters sensitive or harmful words that might surface in both input and output contexts.

```python
# python code
from intel_extension_for_transformers.neural_chat import build_chatbot, PipelineConfig
plugins.safety_checker.enable = True
conf = PipelineConfig(plugins=plugins)
chatbot = build_chatbot(conf)
response = chatbot.predict("Who is lihongzhi?")
print(response)
```

The detailed description about RAG plugin, please refer to [README](./pipeline/plugins/security/README.md)

## Caching

When LLM service encounters higher traffic levels, the expenses related to LLM API calls can become substantial. Additionally, LLM services might exhibit slow response times. Hence, we leverage GPTCache to build a semantic caching plugin for storing LLM responses. Query caching enables the fast path to get the response without LLM inference and therefore improves the chat response time.

```python
# python code
from intel_extension_for_transformers.neural_chat import build_chatbot, PipelineConfig
plugins.cache.enable = True
conf = PipelineConfig(plugins=plugins)
chatbot = build_chatbot(conf)
response = chatbot.predict("Tell me about Intel Xeon Scalable Processors.")
print(response)
```

The detailed description about Caching plugin, please refer to [README](./pipeline/plugins/caching/README.md)


## Inference with Docker

The easiest way of getting started is using the official Docker file. To perform inference, please check [inference with Docker](./docker/inference/README.md). We're on track to release the official Docker containers.



# Advanced Topics

## Optimization

NeuralChat provides typical model optimization technologies, like `Automatic Mixed Precision (AMP)` and `Weight Only Quantization`, to allow user to run a high-througput chatbot.

### Automatic Mixed Precision (AMP)

NeuralChat utilizes Automatic Mixed Precision (AMP) optimization by default when no specific optimization method is specified by the user in the API.
Nevertheless, users also have the option to explicitly specify this parameter, as demonstrated in the following Python code snippet.

```python
# Python code
from intel_extension_for_transformers.neural_chat import build_chatbot, MixedPrecisionConfig
pipeline_cfg = PipelineConfig(optimization_config=MixedPrecisionConfig())
chatbot = build_chatbot(pipeline_cfg)
```

### Weight Only Quantization

Compared to normal quantization like W8A8, weight only quantization is probably a better trade-off to balance the performance and the accuracy. NeuralChat leverages [Intel® Neural Compressor](https://github.com/intel/neural-compressor) to provide efficient weight only quantization.

```python
# Python code
from intel_extension_for_transformers.neural_chat import build_chatbot, PipelineConfig
from intel_extension_for_transformers.transformers import RtnConfig
loading_config = LoadingModelConfig(use_neural_speed=True)
config = PipelineConfig(
    optimization_config=RtnConfig(bits=4, compute_dtype="int8", weight_dtype="int4_fullrange")
)
chatbot = build_chatbot(config)
response = chatbot.predict("Tell me about Intel Xeon Scalable Processors.")
```

### Weight Only Quantization with LLM Runtime
[LLM Runtime](../llm/runtime/graph/README.md) is designed to provide the efficient inference of large language models (LLMs) on Intel platforms in pure C/C++ with optimized weight-only quantization kernels. Applying weight-only quantization with LLM Runtime can yield enhanced performance. However, please be mindful that it might impact accuracy. Presently, we're employing GPTQ for weight-only quantization with LLM Runtime to ensure the accuracy.

```python
# Python code
from intel_extension_for_transformers.neural_chat import build_chatbot, PipelineConfig
from intel_extension_for_transformers.neural_chat.config import LoadingModelConfig
from intel_extension_for_transformers.transformers import RtnConfig
loading_config = LoadingModelConfig(use_neural_speed=True)
config = PipelineConfig(
    optimization_config=RtnConfig(bits=4, compute_dtype="int8", weight_dtype="int4"),
    loading_config=loading_config
)
chatbot = build_chatbot(config)
response = chatbot.predict("Tell me about Intel Xeon Scalable Processors.")
```

## Fine-tuning

NeuralChat supports fine-tuning the pretrained large language model (LLM) for text-generation, summarization, code generation tasks, and even TTS model, for user to create the customized chatbot.

```shell
# Command line
neuralchat finetune --base_model "Intel/neural-chat-7b-v3-1" --config pipeline/finetuning/config/finetuning.yaml
```

```python
# Python code
from intel_extension_for_transformers.neural_chat import finetune_model, TextGenerationFinetuningConfig
finetune_cfg = TextGenerationFinetuningConfig() # support other finetuning config
finetune_model(finetune_cfg)
```

For detailed fine-tuning instructions, please refer to the documentation below.

[NeuralChat Fine-tuning](../examples/finetuning/instruction/README.md)

[Direct Preference Optimization](../examples/finetuning/dpo_pipeline/README.md)

[Reinforcement Learning from Human Feedback](../examples/finetuning/ppo_pipeline/README.md)

[Multi-Modal](../examples/finetuning/multi_modal/README.md)

[How to train Intel/neural-chat-7b-v3-1 on Intel Gaudi2](../examples/finetuning/finetune_neuralchat_v3/README.md)

[Text-To-Speech (TTS) model finetuning](../examples/finetuning/tts/README.md)

And NeuralChat also provides Docker file tailored for easy fine-tuning. Explore details in [finetuning with Docker](../docker/finetuning/README.md).
