Document Indexing
======
1. [Introduction](#introduction)
2. [Get Started](#get-started)

## Introduction

Document indexing serves the purpose of assisting users in parsing locally uploaded files and storing them in a document store for future content retrieval. We have designed two separate indexing methods: sparse retrieval and dense retrieval.

Sparse Retrieval (SR) involves projecting the content into a sparse vector that closely aligns with the vocabulary of the content's language. This can be achieved through traditional Bag-of-Words techniques like TF-IDF or BM25.

On the other hand, Dense Retrieval (DR) encodes the content as one or more dense vectors. Users have the option to specify a local pretrained model or utilize a GPT model from OpenAI to obtain the embeddings of the uploaded content. The choice between sparse retrieval and dense retrieval depends on the specific requirements of the individual.

Our repository currently supports three document stores: `In Memory` and `Elasticsearch` for sparse retrieval, and `Chroma` for dense retrieval. Each document store has its own unique characteristics. The selection of a document store should be based on the maturity of your project, the intended use case, and the technical environment.


|Document store   |Main features       |Platform |
|:----------|:----------|:------------------|
|Elasticsearch  |Sparse retrieval with many tuning options and basic support for dense retrieval. |Haystack|
|In Memory|Simple document store, with no extra services or dependencies. Not recommended for production.  |Haystack                      |                        |
|Chroma    |Focus on dense retrieval. Easy to use. Lightwieght and fast for retrieval.    |LangChain|

The support for other document stores will be available soon.

Right now, we support the user to upload the file in the PDF format and jsonl format. After the indexing work, the user can easily edit the local document store to add or delete a specific file. 

## Get Started

### Sparse Indexing

When it comes to sparse indexing, the process of parsing a local file into the desired document store is straightforward for users. They simply need to provide the file path using the `--file_path` parameter and choose an appropriate local document store using the `--store parameter`.

However, it's important to mention that the `In Memory` method does not support local database storage. It requires users to perform document processing every time they use it, as the documents are not persistently stored.

 ```bash 
python doc_index.py --file_path "xxx" --output_path "xxx" --embedding_method sparse --store Elasticsearch
 ```

### Dense Indexing
When it comes to dense indexing, users have the flexibility to choose their preferred pretrained encoder model for the process. In the given use case, we utilize the `instructor-large` model with the `HuggingFaceInstructEmbeddings` API. Users can select a suitable model from the [text embedding benchmark leaderboard](https://huggingface.co/spaces/mteb/leaderboard). We support both local models and models available through the HuggingFace library. To use a specific model, users can provide the model name.

Alternatively, users can also utilize GPT models from OpenAI. To incorporate a GPT model into the process, minor adjustments need to be made to the following code:
 ```python
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
 ```

The user can start the dense indexing with,
 ```bash 
python doc_index.py --file_path "xxx" --output_path "xxx" --embedding_model hkunlp/instructor-large --embedding_method dense --store Chroma
 ```