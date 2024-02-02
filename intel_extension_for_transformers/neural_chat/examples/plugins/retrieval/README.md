This example demonstrates how to enhance the generation ability of your chatbot with your local documents within few lines of codes.

# Introduction
The popularity of applications like ChatGPT has attracted many users seeking to address everyday problems. However, some users have encountered a challenge known as "model hallucination," where LLMs generate incorrect or nonexistent information, raising concerns about content accuracy. This example introduce our solution to build a retrieval-based chatbot. Though few lines of code, our api could help the user build a local reference ddatabase to enhance the accuracy of the generation results.

Before deploying this example, please follow the instructions in the [README](../../README.md) to install the necessary dependencies.

# Usage
The Neural Chat API offers an easy way to create and utilize chatbot models while integrating local documents. Our API simplifies the process of automatically handling and storing local documents in a document store. The user an download the [Intel 2022 Annual Report](https://d1io3yog0oux5.cloudfront.net/_897efe2d574a132883f198f2b119aa39/intel/db/888/8941/file/412439%281%29_12_Intel_AR_WR.pdf) for a test.

## Import the module and set the retrieval config:

```python
from intel_extension_for_transformers.neural_chat import PipelineConfig
from intel_extension_for_transformers.neural_chat import plugins
plugins.retrieval.enable=True
plugins.retrieval.args["input_path"]="./Annual_Report.pdf"
config = PipelineConfig(plugins=plugins)
```

## Build the chatbot and interact with the chatbot:

```python
from intel_extension_for_transformers.neural_chat import build_chatbot
chatbot = build_chatbot(config)
response = chatbot.predict("What is IDM 2.0?")
```

## Run the complete code
```shell
python retrieval_chat.py
```

## Performance acceleration on Intel® Xeon SPR
You can utilize the following script to execute the code on Intel® Xeon SPR processor to accelerate the inference.
```bash
conda install jemalloc gperftools -c conda-forge -y
bash run_retrieval_on_cpu.sh
```
