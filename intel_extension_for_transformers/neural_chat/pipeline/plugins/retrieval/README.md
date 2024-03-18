<div align="center">
<h1>Retrieval</h3>
<div align="left">

# Introduction
Large language models (LLMs) have shown exceptional performance in various natural language processing tasks, establishing them as essential tools for understanding and generating language. These models absorb extensive world knowledge from their vast training datasets, enabling them to produce fluent and coherent responses to user queries without external data resources. However, despite the remarkable breadth of knowledge large language models gain during training, they face limitations in accessing up-to-date information and certain domain-specific data. This limitation can lead to a significant concern: the tendency of large language models to 'hallucinate' or create fictitious content in their responses.

This hallucination problem primarily arises from two factors: (1) LLMs are predominantly trained using data from the Internet, which limits their exposure to specific, domain-focused information; and (2) LLMs mainly rely on the training corpus for information extraction. These models remain unaware of events occurring post-training, which can be particularly problematic for topics that change daily. Two methods are recognized as effective method for model hallucination problems, [Finetuning the LLM on task-specific datasets](https://arxiv.org/abs/2311.08401) and [Retrieval-Augmented Generation (RAG)](https://arxiv.org/abs/2212.10560). However, finetuning a LLM is impractical for most users due to it requires high-quality datasets, labor-intensive data annotation, and substantial computational resources. Also, it is challenging to collect and maintain an extensive, up-to-date knowledge corpus. Therefore, we propose an economically efficient alternative based on RAG. It retrieves relevant documents from a local database to serve as the reference to enhance the accuracy and reliability of the generated results.

Inspired by the prevent chatbot framework [langchain](https://github.com/langchain-ai/langchain), [Llama-Index](https://github.com/run-llama/llama_index) and [haystack](https://github.com/deepset-ai/haystack), our NeuralChat API offers an easy way to create and utilize chatbot models while integrating RAG. Our API provides an easy to use extension for langchain users as well as a convenient deployment code for the general user. Without too much learning effort, the user can build their own RAG-based chatbot with their documents. The details about our langchain extension feature could be see [here](#langchain-extension).

Currently, we concentrate on [dense retrieval](https://medium.com/@aikho/deep-learning-in-information-retrieval-part-ii-dense-retrieval-1f9fecb47de9) to construct the RAG pipeline. The dense retrieval will return the documents that share the similar semantic expression with the candidate queries instead of the keywords expression, which is more suitable for the long-context application scenario.

The embedding model plays a crucial factor to influence the retrieval accuracy. We have already provided support for a wide range of open-released pre-trained embedding models featured on the [HuggingFace text embedding leaderboard](https://huggingface.co/spaces/mteb/leaderboard). Users can conveniently choose an embedding model in two ways: they can either specify the model by its name on HuggingFace or download a model and save it under the default name. Below is a list of some supported embedding models available in our plugin. Users can select their preferred embedding model based on various factors such as model size, embedding dimensions, maximum sequence length, and average ranking score.
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

> Due to the recent code refactorization of `sentence-transformers` will impact the operation behaviour of the embedding models, please check and install the latest `sentence-transformers` from source!  

This plugin streamlines three key processes: parsing documents, identifying user intentions, and fetching relevant information. Initially, the `Agent_QA` sets itself up by building a local database from the data at input_path. In the midst of a conversation, when a user poses a question, it first goes through the `IntentDetector`. This step is crucial to figure out if the user is just making casual conversation or looking for specific information. If the IntentDetector concludes that the user is seeking an answer, it triggers the `retrieval` process. This involves scouring the database with the user's question to find pertinent information. The information thus obtained forms the basis for crafting responses with the help of Large Language Models (LLMs).

To ensure a smooth experience, we've made sure this plugin is compatible with common file formats like xlsx, csv, and json/jsonl. It's important to note that these files need to follow a specific structure for optimal functioning.
|  File Type   | Predefined Structure  |
|  :----:  | :----:  |
| xlsx  | ['Questions', 'Answers']<br>['question', 'answer', 'link']<br>['context', 'link'] |
| csv  | ['question', 'correct_answer'] |
| json/jsonl  | {'content':xxx, 'link':xxx}|
| txt  | No format required |
| html  | No format required |
| markdown  | No format required |
| word  | No format required |
| pdf  | No format required |

# Usage
Before using RAG in NeuralChat, please install the necessary dependencies in [requirements.txt](https://github.com/intel/intel-extension-for-transformers/blob/main/intel_extension_for_transformers/neural_chat/pipeline/plugins/retrieval/requirements.txt) to avoid the import errors. The most convenient way to use is this plugin is via our `build_chatbot` api as introduced in the [example code](https://github.com/intel/intel-extension-for-transformers/tree/main/intel_extension_for_transformers/neural_chat/examples/plugins/retrieval). The user could refer to it for a simple test. 

We support multiple file formats for retrieval, including unstructured file formats such as pdf, docx, html, txt, and markdown, as well as structured file formats like jsonl/json, csv, xlsx. For structured file formats, they must adhere to predefined structures. We also support to upload the knowledge base via a http web link.

In the case of jsonl files, they should be formatted as dictionaries, such as: {'content':xxx, 'link':xxx}. The support for xlsx files is specifically designed for Question-Answer (QA) tasks. Users can input QA pairs for retrieval. Therefore, the table's header should include items labeled as "Question" and "Answer". The reference files could be found [here](https://github.com/intel/intel-extension-for-transformers/tree/main/intel_extension_for_transformers/neural_chat/assets/docs).

## Import the module and set the retrieval config:
> The user can download the [Intel 2022 Annual Report](https://d1io3yog0oux5.cloudfront.net/_897efe2d574a132883f198f2b119aa39/intel/db/888/8941/file/412439%281%29_12_Intel_AR_WR.pdf) for a quick test.

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
response = chatbot.predict("What is IDM 2.0?")
```

Checkout the full example [retrieval_chat.py](../../../examples/retrieval/retrieval_chat.py) and have a try!

# Parameters
Users have the flexibility to tailor the retrieval configuration to meet their individual needs and adapt to their local files. To customize a particular aspect of the retrieval plugin, you can adjust its settings as follows:
```python
plugins.retrieval.args["xxx"]=xxx
```
Below are the description for the available parameters in `agent_QA`,

|  Parameters   |  Type | Description| Options|
|  ----  | ----  | --| --|
| vector_database  | str | The vector database for constructing the knowledge base. |"Chroma", "Qdrant"|
| input_path   | str | The path of the file/folder/link of the content to formulate the knowledge base |-|
| embedding_model  | str | The name or path for the text embedding model |-|
| response_template  | str | Default response when there is no available relevant documents for RAG |-|
| mode  | str | The RAG behavior for different use case. Please check [here](#rag-mode) |"accuracy", "general"|
| retrieval_type   | str | The type of the retriever. Please check [here](#retrievers) for more details  | "default", "child_parent", "bm25"|
| process  | bool | Whether to split the long documents into small chucks. The size of each chuck is defined by `max_chuck_size` and `min_chuck_size`|True, False|
| max_chuck_size  | int | The max token length for a single chuck in the knowledge base |-|
| min_chuck_size  | int | The min token length for a single chuck in the knowledge base |-|
| append  | bool | Whether the new knowledge will be append to the existing knowledge base or directly load the existing knowledge base |True, False|
| polish  | bool | Whether to polish the input query before processing |True, False|
| enable_rerank   | bool | Whether to enable retrieval then rerank pipeline |True, False|
| reranker_model   | str | The name of the reranker model from the Huggingface or a local path |-|
| top_n   | int | The return number of the reranker model |-|

More retriever- and vectorstore-related parameters please check [here](#langchain-extension)

# RAG Mode
Our system offers two distinct modes for the Retrieval-Augmented Generation (RAG) feature, catering to different user expectations: "accuracy" and "general." These modes are designed to accommodate various application scenarios.

In "general" mode, the system primarily utilizes the output of the `IntentDetector` to determine the appropriate response prompt. If the predicted intent of the user's query is "chitchat," the system engages in a casual conversation. For other intents, it crafts a response augmented by retrieval results. This mode leverages the Large Language Model's (LLM's) inherent capabilities to predict user intent and generate relevant responses. However, it may occasionally misinterpret user intent, leading to reliance on the LLM's inherent knowledge for response generation, which could result in inaccuracies or model hallucination issues.

Conversely, "accuracy" mode combines the `IntentDetector`'s output with retrieval results to enhance the accuracy of intent prediction. We implement a retrieval threshold to balance free generation with reliance on relevant documents. In this mode, the system will first search for relevant content to support the response. Casual conversation ("chitchat") only occurs if there are no relevant documents and the intent is determined as such. This approach helps mitigate model hallucination problems but may limit the LLM's free generation capacity.

Users are encouraged to choose the RAG mode that best suits their specific needs and application scenario.

# Langchain Extension
To fully leverage the capabilities of our mutual chatbot platform, we have developed a comprehensive range of langchain-based extension APIs. These enhancements include advanced retrievers, embedding models, and vector stores, all designed to expand the functionality of the original langchain API. Our goal with these additions is to enrich user experience and provide a more robust and versatile chatbot platform.

## Vector Stores

### Chroma
[Chroma](https://docs.trychroma.com/getting-started) stands out as an AI-native, open-source vector database, placing a strong emphasis on boosting developer productivity and satisfaction. It's available under the Apache 2.0 license. Initially, the original Chroma API within langchain was designed to accept settings only once, at the chatbot's startup. This approach lacked flexibility, as it didn't allow users to modify settings post-initialization. To address this limitation, we've revamped the Chroma API. Our updated version introduces enhanced vector store operations, enabling users to adjust and fine-tune their settings even after the chatbot has been initialized, offering a more adaptable and user-friendly experience.

The user can select Chroma as the vectorstore for RAG with:
```python
plugins.retrieval.args["vector_database"]="Chroma"
```
Our Chroma API is easy to use and can be generalized to langchain platform. For a quick Chroma configuration, the user can directly set the parameters following the same step for [agent_QA](#parameters). Some of parameters for Chroma share the same value with agent_QA. The extra parameters for Chroma are:
|  Parameters   |  Type | Description| Options|
|  ----  | ----  | --| --|
| collection_name  | str | The collection name for the local Chroma database instance. |-|
| persist_directory   | str | The path for saving the knowledge base. |-|
| collection_metadata   | dict | Collection configurations. Can set the retrieval distance type and indexing structure. |-|

For the langchain users, it can be easily imported and used by replacing the origin Chroma API in langchain.
```python
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_core.vectorstores import VectorStoreRetriever
from intel_extension_for_transformers.langchain_community.vectorstores import Chroma
retriever = VectorStoreRetriever(vectorstore=Chroma(...))
retrievalQA = RetrievalQA.from_llm(llm=HuggingFacePipeline(...), retriever=retriever)
```
More independent langchain-based examples can be found [here](https://python.langchain.com/docs/integrations/vectorstores/chroma).

### Qdrant
[Qdrant](https://qdrant.tech/documentation/) is a state-of-the-art vector similarity search engine, designed for production-ready services. It features an easy-to-use API that enables the storage, search, and management of points - vectors that come with additional payload data. Qdrant stands out for its advanced filtering capabilities, making it ideal for a wide range of uses such as neural network or semantic-based matching, faceted search, and other similar applications.

Originally, the Qdrant API within langchain was set up to allow configuration only once, at the time of the chatbot's initialization. This setup limited flexibility, as it didn't permit users to modify settings after the initial setup. Recognizing this limitation, we have redeveloped the Qdrant API. Our enhanced version offers expanded vector store operations, providing users with the ability to adjust and refine their settings post-initialization, thereby delivering a more adaptable and user-friendly experience.

The user can select Qdrant as the vectorstore for RAG with:
```python
plugins.retrieval.args["vector_database"]="Qdrant"
```
Our Qdrant API is easy to use and can be generalized to langchain platform. For a quick Qdrant configuration, the user can directly set the parameters following the same step for [agent_QA](#parameters). Some of parameters for Qdrant share the same value with agent_QA. The extra parameters for Qdrant are:
|  Parameters   |  Type | Description| Options|
|  ----  | ----  | --| --|
| collection_name  | str | The collection name for the local Qdrant database instance. |-|
| location  | str | If `:memory:` - use in-memory Qdrant instance. If `str` - use it as a `url` parameter. If `None` - fallback to relying on `host` and `port` parameters.  |-|
| url   | str | Either host or str of "Optional[scheme], host, Optional[port], Optional[prefix]" |-|
| host   | str | Host name of Qdrant service. If url and host are None, set to 'localhost'. |-|
| persist_directory   | str | Path in which the vectors will be stored while using local mode.|-|

For the langchain users, it can be easily imported and used by replacing the origin Qdrant API in langchain.
```python
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_core.vectorstores import VectorStoreRetriever
from intel_extension_for_transformers.langchain_community.vectorstores import Qdrant
retriever = VectorStoreRetriever(vectorstore=Qdrant(...))
retrievalQA = RetrievalQA.from_llm(llm=HuggingFacePipeline(...), retriever=retriever)
```
More independent langchain-based examples can be found [here](https://python.langchain.com/docs/integrations/vectorstores/qdrant).

## Retrievers
Retrievers play a crucial role for RAG. They are responsible for implementing the basic retrieval configuration and accessing the vectorstore using the specified retrieval method and settings. Currently, we offer two types of retrievers: `VectorStoreRetriever` and `ChildParentRetriever`.

We've chosen VectorStoreRetriever as the default retriever. This decision aligns the retrieval process seamlessly with langchain’s functionality. The VectorStoreRetriever is designed to efficiently handle vectorstore operations, ensuring optimal retrieval performance. Meanwhile, the ChildParentRetriever offers a special solution for the long-context scenario.

Our approach ensures that users have access to versatile and effective retrieval tools, tailored to a variety of requirements and preferences within the system.

### VectorStoreRetriever
We've maintained most of the retrieval behaviors consistent with langchain. The user can select this retriever by:
```python
plugins.retrieval.args["retrieval_type"]="default"
```

The basic parameters for `VectorStoreRetriever` are:
|  Parameters   |  Type | Description| Options|
|  ----  | ----  | --| --|
| search_type  | str | Type of search to perform. |"mmr", "similarity_score_threshold", and "similarity"|
| search_kwargs  | dict | Keyword arguments to pass to the search function.|-|

The user can set the parameters for the retriever by: 
```python
plugins.retrieval.args["search_type"]=xxx
plugins.retrieval.args["search_kwargs"]=xxx
```

If "search_type"="similarity":
>search_kwargs={"k":xxx}

"k" is the number of the returned most similar documents.

If "search_type"="mmr":
>search_kwargs={"k":xxx, "fetch_k":xxx, "lamabda_mult":xxx}

"k" is the number of the returned most similar documents. "fetch_k" is the number of Documents to fetch to pass to MMR algorithm. "Lamabda_mult" is a number between 0 and 1 that determines the degree of diversity among the results with 0 corresponding to maximum diversity and 1 to minimum diversity. Defaults to 0.5.

If "search_type"="similarity_score_threshold":
>search_kwargs={"k":xxx, "score_threshold":xxx}

"k" is the number of the returned most similar documents. "score_threshold" is the similar score threshold for the retrieved documents.

### ChildParentRetriever
We've specifically designed this retriever to address challenges in long-context retrieval scenarios. Commonly, in many applications, the documents being retrieved are lengthier than the user's query. This discrepancy leads to an imbalance in context information between the query and the documents, often resulting in reduced retrieval accuracy. The reason is that the documents typically contain a richer semantic expression compared to the brief user query.

An ideal solution would be to segment the user-uploaded documents for the RAG knowledgebase into suitably sized chunks. However, this approach is not always feasible due to the lack of consistent guidelines for automatically and accurately dividing the context. Too short a division can result in partial, contextually incomplete answers to user queries. Conversely, excessively long segments can significantly lower retrieval accuracy.

To navigate this challenge, we've developed a unique solution involving the `ChildParentRetriever` to optimize the RAG process. Our strategy involves initially splitting the user-uploaded files into larger chunks, termed 'parent chunks', to preserve the integrity of each concept. Then, these parent chunks are further divided into smaller 'child chunks'. Both child and parent chunks are interconnected using a unique identification ID. This approach enhances the likelihood and precision of matching the user query with a relevant, concise context chunk. When a highly relevant child chunk is identified, we use the ID to trace back to its parent chunk. The context from this parent chunk is then utilized in the RAG process, thereby improving the overall effectiveness and accuracy of retrieval.

The user can select this retriever by:
```python
plugins.retrieval.args["retrieval_type"]="child_parent"
```

Most parameters for `ChildParentRetriever` are will be automatically set by `agent_QA`. The user only needs to decide the `search_type` and `search_kwargs`.
|  Parameters   |  Type | Description| Options|
|  ----  | ----  | --| --|
| search_type  | str | Type of search to perform. |"mmr", and "similarity"|
| search_kwargs  | dict | Keyword arguments to pass to the search function.|-|

The user can set the parameters for the retriever by: 
```python
plugins.retrieval.args["search_type"]=xxx
plugins.retrieval.args["search_kwargs"]=xxx
```
If "search_type"="similarity":
>search_kwargs={"k"=xxx}

If "search_type"="mmr":
>search_kwargs={"k"=xxx, "fetch_k"=xxx, "lamabda_mult"=xxx}

"k" is the number of the returned most similar documents. "fetch_k" is the number of Documents to fetch to pass to MMR algorithm. "Lamabda_mult" is a number between 0 and 1 that determines the degree of diversity among the results with 0 corresponding to maximum diversity and 1 to minimum diversity. Defaults to 0.5.

This new retriever is also available for langchain users. Below is a toy example that using our `ChildParentRetriever` in the langchain framework:
```python
from intel_extension_for_transformers.langchain_community.retrievers import ChildParentRetriever
from langchain.vectorstores import Chroma
retriever = ChildParentRetriever(vectorstore=Chroma(documents=child_documents), parentstore=Chroma(documents=parent_documents), search_type=xxx, search_kwargs={...})
docs=retriever.get_relevant_documents("Intel")
```
