This example demonstrates how to enhance the generation ability of your chatbot with your local documents within few lines of codes.

# Introduction
Large Language Models (LLMs) have demonstrated remarkable performance in various Natural Language Processing (NLP) tasks. Compared to earlier pretrained models, LLMs can produce better results on tasks without fine-tuning, reducing the cost of use. The popularity of applications like ChatGPT has attracted many users seeking to address everyday problems. However, some users have encountered a challenge known as "model hallucination," where LLMs generate incorrect or nonexistent information, raising concerns about content accuracy.

To improve the accuracy of generated content, two approaches can be considered: expanding the training data or utilizing an external database. Expanding the training data is impractical due to the time and effort required to train a high-performance LLM. It's challenging to collect and maintain an extensive, up-to-date knowledge corpus. Therefore, we propose an economically efficient alternative: leveraging relevant documents from a local database during content generation. These retrieved documents will be integrated into the input prompt of the LLM to enhance the accuracy and reliability of the generated results.

Before deploying this example, please follow the instructions in the [README](../../README.md) to install the necessary dependencies.

# Usage
The Neural Chat API offers an easy way to create and utilize chatbot models while integrating local documents. Our API simplifies the process of automatically handling and storing local documents in a document store. We provide support for two retrieval methods:
1. Dense Retrieval: This method is based on document embeddings, enhancing the accuracy of retrieval. Learn more about [here](https://medium.com/@aikho/deep-learning-in-information-retrieval-part-ii-dense-retrieval-1f9fecb47de9) (based on the document embedding) 
2. Sparse Retrieval: Using TF-IDF, this method efficiently retrieves relevant information. Explore this approach in detail [here](https://medium.com/itnext/deep-learning-in-information-retrieval-part-i-introduction-and-sparse-retrieval-12de0423a0b9).

## Import the module and set the retrieval config:
The user can download the [Intel 2022 Annual Report](https://d1io3yog0oux5.cloudfront.net/_897efe2d574a132883f198f2b119aa39/intel/db/888/8941/file/412439%281%29_12_Intel_AR_WR.pdf) for a test.

```python
from intel_extension_for_transformers.neural_chat import PipelineConfig
from intel_extension_for_transformers.neural_chat import plugins
plugins.retrieval.enable=True
plugins.retrieval.args["retrieval_input_path"]="./Annual_report.pdf"
config = PipelineConfig(plugins=plugins)
```

## Build the chatbot and interact with the chatbot:

```python
from intel_extension_for_transformers.neural_chat import build_chatbot
chatbot = build_chatbot(config)
response = chatbot.predict("What is IDM 2.0?")
```

## Deploying through shell comand:
```shell
python retrieval_chat.py
```

# Parameters
The user can costomize the retrieval parameters to meet the personal demmads for better catering the local files. Please check [here](https://github.com/intel/intel-extension-for-transformers/blob/main/intel_extension_for_transformers/neural_chat/pipeline/plugins/retrievals/retrievals.py) for more details.


