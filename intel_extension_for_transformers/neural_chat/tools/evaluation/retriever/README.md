# Retrieval Evaluation

## 1. Introduction
We provide a script to evaluate the performance of the retrieval. We use two metrics: MRR (Mean reciprocal rank) and Hit (Hit Ratio). 
* **MRR** is an internationally accepted mechanism for evaluating search algorithms. MRR emphasizes the position of ground truth in the retrieval list, the higher it is, the better. 
* **Hit** emphasizes the accuracy of retrieval, that is, whether the ground truth is included in the retrieval items. The higher, the better. 

## 2. Installation
Please ensure the installation of requirements for NeuralChat and retrieval plugin by the following commands.
```
git clone https://github.com/intel/intel-extension-for-transformers.git
cd intel-extension-for-transformers/intel_extension_for_transformers/neural_chat
pip install -r requirements.txt
cd pipeline/plugins/retrieval
pip install -r requirements.txt
```

## 3. Evaluate Retrieval
You can evaluate the retrieval performance by the following commands.
```
cd intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/evaluation/retriever
python evaluate_retrieval.py \
--index_file_jsonl_path /path/to/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/evaluation/data_augmentation/candidate_context.jsonl \
--query_file_jsonl_path /path/to/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/evaluation/data_augmentation/example.jsonl
```

**Some Important Arguments**:
- `index_file_jsonl_path`: path of JSON data including candidate context where each line is a dict like this:```{"context": List[str]}```. See [candidate_context.jsonl](https://github.com/intel/intel-extension-for-transformers/blob/master/intel_extension_for_transformers/neural_chat/tools/evaluation/data_augmentation/candidate_context.jsonl) for a data file.
- `query_file_jsonl_path`: path of JSON data including queries and positives where each line is a dict like this:```{"query": str, "pos": List[str]}```. See [example.jsonl](https://github.com/intel/intel-extension-for-transformers/blob/master/intel_extension_for_transformers/neural_chat/tools/evaluation/data_augmentation/example.jsonl) for a data file.
- `vector_database`: The vector database for constructing the knowledge base. The default value is "Chroma". The other option is "Qdrant".
- `embedding_model`: The name or path for the text embedding model. The default value is "BAAI/bge-base-en-v1.5". Other options are "BAAI/bge-large-en-v1.5", "thenlper/gte-large", "infgrad/stella-base-en-v2", "thenlper/gte-base", "intfloat/e5-large-v2", "hkunlp/instructor-xl", and "hkunlp/instructor-large".
- `retrieval_type`: The type of the retriever. The default value is "default". The other options are "child_parent" and "bm25".
- `search_type`: Type of search to perform. The default value is "similarity". The other options are "mmr" and "similarity_score_threshold".

## 4. Result
The results include Top 1 and Top 5 of MRR and HR respectively.
```
{'MRR@1': 0.7, 'MRR@5': 0.72, 'Hit@1': 0.7, 'Hit@5': 0.8}
```
