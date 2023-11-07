<div align="center">
<h1>Retrieval</h3>
<div align="left">

# Introduction
Large Language Models (LLMs) have demonstrated remarkable performance in various Natural Language Processing (NLP) tasks. Compared to earlier pretrained models, LLMs can produce better results on tasks without fine-tuning, reducing the cost of use. The popularity of applications like ChatGPT has attracted many users seeking to address everyday problems. However, some users have encountered a challenge known as "model hallucination," where LLMs generate incorrect or nonexistent information, raising concerns about content accuracy.

To improve the accuracy of generated content, two approaches can be considered: expanding the training data or utilizing an external database. Expanding the training data is impractical due to the time and effort required to train a high-performance LLM. It's challenging to collect and maintain an extensive, up-to-date knowledge corpus. Therefore, we propose an economically efficient alternative: leveraging relevant documents from a local database during content generation. These retrieved documents will be integrated into the input prompt of the LLM to enhance the accuracy and reliability of the generated results.

The Neural Chat API offers an easy way to create and utilize chatbot models while integrating local documents. Our API simplifies the process of automatically handling and storing local documents in a document store. We provide support for two retrieval methods:
1. Dense Retrieval: This method is based on document embeddings, enhancing the accuracy of retrieval. Learn more about [here](https://medium.com/@aikho/deep-learning-in-information-retrieval-part-ii-dense-retrieval-1f9fecb47de9).
2. Sparse Retrieval: Using TF-IDF, this method efficiently retrieves relevant information. Explore this approach in detail [here](https://medium.com/itnext/deep-learning-in-information-retrieval-part-i-introduction-and-sparse-retrieval-12de0423a0b9).

We have already provided support for a wide range of pre-trained embedding models featured on the [HuggingFace text embedding leaderboard](https://huggingface.co/spaces/mteb/leaderboard). Users can conveniently choose an embedding model in two ways: they can either specify the model by its name on HuggingFace or download a model and save it under the default name. Below is a list of some supported embedding models available in our plugin. Users can select their preferred embedding model based on various factors such as model size, embedding dimensions, maximum sequence length, and average ranking score.
|  Model   | Model Size (GB)  |Embedding Dimensions  |Max Sequence Length  |Average Ranking Score  |
|  :----:  | :----:  | :----:  | :----: |:----: |
| [bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)  | 1.34 |1024  |512  |64.23|
| [bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5)  | 0.44 |768  |512  |63.55|
| [	gte-large](https://huggingface.co/thenlper/gte-large)  | 0.67 |1024  |512  |63.13|
| [stella-base-en-v2](https://huggingface.co/infgrad/stella-base-en-v2)  | 0.22 |768  |512 |62.61|
| [gte-base](https://huggingface.co/thenlper/gte-base)  | 0.44 |768  |512  |62.39|
| [	e5-large-v2](https://huggingface.co/intfloat/e5-large-v2)  | 1.34 |1024  |512  |62.25|
| [instructor-xl](https://huggingface.co/hkunlp/instructor-xl)  | 4.96 |768  |512  |61.79|
| [instructor-large](https://huggingface.co/hkunlp/instructor-large)  | 1.34 |768  |512  |61.59|

In addition, our plugin seamlessly integrates the online embedding model, Google Palm2 embedding. To set up this feature, please follow the [Google official guideline](https://developers.generativeai.google/tutorials/embeddings_quickstart) to obtain your API key. Once you have your API key, you can activate the Palm2 embedding service by setting the `embedding_model` parameter to 'Google'.

The workflow of this plugin consists of three main operations: document indexing, intent detection, and retrieval. The `Agent_QA` initializes itself using the provided `input_path` to construct a local database. During a conversation, the user's query is first passed to the `IntentDetector` to determine whether the user intends to engage in chitchat or seek answers to specific questions. If the `IntentDetector` determines that the user's query requires an answer, the retriever is activated to search the database using the user's query. The documents retrieved from the database serve as reference context in the input prompt, assisting in generating responses using the Large Language Models (LLMs). 

# Usage
The most convenient way to use is this plugin is via our `build_chatbot` api as introduced in the [example code](https://github.com/intel/intel-extension-for-transformers/tree/main/intel_extension_for_transformers/neural_chat/examples/plugins/retrieval). The user could refer to it for a simple test. 

We support multiple file formats for retrieval, including unstructured file formats such as pdf, docx, html, txt, and markdown, as well as structured file formats like jsonl and xlsx. For structured file formats, they must adhere to predefined structures.

In the case of jsonl files, they should be formatted as dictionaries, such as: {"doc": xxx, "doc_id": xxx}. The support for xlsx files is specifically designed for Question-Answer (QA) tasks. Users can input QA pairs for retrieval. Therefore, the table's header should include items labeled as "Question" and "Answer". The reference files could be found [here](https://github.com/intel/intel-extension-for-transformers/tree/main/intel_extension_for_transformers/neural_chat/assets/docs).

## Import the module and set the retrieval config:
The user can download the [Intel 2022 Annual Report](https://d1io3yog0oux5.cloudfront.net/_897efe2d574a132883f198f2b119aa39/intel/db/888/8941/file/412439%281%29_12_Intel_AR_WR.pdf) for a quick test.

```python
from intel_extension_for_transformers.neural_chat import PipelineConfig
from intel_extension_for_transformers.neural_chat import plugins
plugins.retrieval.enable=True
plugins.retrieval.args["input_path"]="./Annual_report.pdf"
config = PipelineConfig(plugins=plugins)
```

## Build the chatbot and interact with the chatbot:

```python
from intel_extension_for_transformers.neural_chat import build_chatbot
chatbot = build_chatbot(config)
response, link = chatbot.predict("What is IDM 2.0?")
```

Checkout the full example [retrieval_chat.py](../../../examples/retrieval/retrieval_chat.py) and have a try!

# Parameters
The user can costomize the retrieval parameters to meet the personal demmads for better catering the local files. The user can set the specific parameter by plugins.retrieval.args["xxx"]. Below the description of each available parameters,

```python
persist_dir [str]: The local path to save the processed database. Default to "./output".

process [bool]: Select to process the too long document into small chucks. Default to "True".

input_path [str]: The user local path to a file folder or a specific file path. The code itself will check the path is a folder or a file. If it is a folder, the code will process all the files in the given folder. If it is a file, the code will prcess this single file.

embedding_model [str]: the user specific document embedding model for dense retrieval. The user could selecte a specific embedding model from "https://huggingface.co/spaces/mteb/leaderboard". Default to "BAAI/bge-base-en-v1.5". 

max_length [int]: The max context length in the processed chucks. Should be combined with "process". Default to "512".

retrieval_type [str]: Select a method for retrieval from "dense" or "sparse". Default to "dense".

document_store [str]: Considering the sparse retrieval needs to load the data into memory. We provide "InMemoryDocumentStore" and "ElasticsearchDocumentStore" for manage the memory efficiency for sparse retrieval. Default to "None" for using dense retrieval.
    
top_k [int]: The number of the retrieved documents. Default to "1".

search_type [str]: Select a ranking method for dense retrieval from "mmr", "similarity" and "similarity_score_threshold". "similarity" will return the most similar docs to the input query. "mmr" will return ranking the docs using the maximal marginal relevance method. "similarity_score_threshold" will return the mosy similar docs that also meet the threshold. Deault to "mmr".

search_kwargs [dict]: Used by dense retrieval. Should be in the same format like {"k":1, "fetch_k":5}. "k" is the amount of documents to return. "score_threshold" is the minimal relevance threshold for "similarity_score_threshold" search. "lambda_mult" is the diversity of results returned by "mmr". "fetch_k" determines the amount of documents to pass to the "mmr" algorithm. Default to {"k":1, "fetch_k":5}.

append [bool]: Decide to append the local database or not. If append=True, the uploaded files will be continuously written into the database. If append=False, the existing database will be loaded.

index_name [str]: The index name for ElasticsearchDocumentStore.
```
