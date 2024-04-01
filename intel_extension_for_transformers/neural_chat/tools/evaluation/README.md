# Retrieval and Rag Benchmark

## 1. Introduction
We provide scripts of the benchmark of Retrieval and Rag. For data augmentation, please go to [Retrieval Data Augmentation](https://github.com/intel/intel-extension-for-transformers/blob/master/intel_extension_for_transformers/neural_chat/tools/evaluation/data_augmentation).

## 2. Retrieval Benchmark
### Installation
Please ensure the installation of requirements for NeuralChat and retrieval plugin by the following commands.
```
git clone https://github.com/intel/intel-extension-for-transformers.git
cd intel-extension-for-transformers/intel_extension_for_transformers/neural_chat
pip install -r requirements.txt
cd pipeline/plugins/retrieval
pip install -r requirements.txt
```

### Benchmark
You can run retrieval benchmark by the following commands.
```
cd intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/evaluation/retriever
bash retrieval_benchmark.sh \
--index_file_jsonl_path=/path/to/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/evaluation/data_augmentation/candidate_context.jsonl \
--query_file_jsonl_path=/path/to/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/evaluation/data_augmentation/example.jsonl
--vector_database=Chroma \
--embedding_model=<embedding model name or path> \
--llm_model=<llm model name or path> \
--reranker_model=<reranker model name or path>
```
**Some Important Arguments**:
- `index_file_jsonl_path`: path of JSON data including candidate context where each line is a dict like this:```{"context": List[str]}```. See [candidate_context.jsonl](https://github.com/intel/intel-extension-for-transformers/blob/master/intel_extension_for_transformers/neural_chat/tools/evaluation/data_augmentation/candidate_context.jsonl) for a data file.
- `query_file_jsonl_path`: path of JSON data including queries and positives where each line is a dict like this:```{"query": str, "pos": List[str]}```. See [example.jsonl](https://github.com/intel/intel-extension-for-transformers/blob/master/intel_extension_for_transformers/neural_chat/tools/evaluation/data_augmentation/example.jsonl) for a data file.
- `vector_database`: The vector database for constructing the knowledge base.
- `embedding_model`: The name or path for the text embedding model. The default value is "BAAI/bge-base-en-v1.5". Other options are "BAAI/bge-large-en-v1.5", "thenlper/gte-large", "infgrad/stella-base-en-v2", "thenlper/gte-base", "intfloat/e5-large-v2", "hkunlp/instructor-xl", and "hkunlp/instructor-large".
- `llm_model`: The name or path for the LLM model.
- `reranker_model`: The name or path for the reranker model.
- `retrieval_type`: The type of the retriever. The default value is "default". The other options are "child_parent" and "bm25".
- `polish`: Whether to polish the input query before processing. The default value is False.
- `search_type`: Type of search to perform. The default value is "similarity". The other options are "mmr" and "similarity_score_threshold".
- `k`: The number of the returned most similar documents. The default value is 1.
- `fetch_k`: The number of Documents to fetch to pass to MMR algorithm. The default value is 5.
- `score_threshold`: The similar score threshold for the retrieved documents. The default value is 0.3.
- `top_n`: The return number of the reranker model. The default value is 1.
- `enable_rerank`: Whether to enable retrieval then rerank pipeline. The default value is False.

**Result**:
The result will include all parameter values and MRR (Mean reciprocal rank) and Hit (Hit Ratio) values.
```
|  Parameter & Result  | Value  |
|  :----:  | :----:  |
| 'index_file_jsonl_path'  | '/path/to/candidate_context.jsonl' |
| 'query_file_jsonl_path'  | '/path/to/example.jsonl' |
| 'vector_database'  | 'Chroma'|
| 'embedding_model' | '/path/to/bge-large-en-v1.5' |
| 'retrieval_type' | 'default' |
| 'polish' | False |
| 'search_type' | 'similarity' |
| 'llm_model' | '/path/to/neural-chat-7b-v3-1/' |
| 'k' | 1 |
| 'fetch_k' | 5 |
| 'score_threshold' | 0.3 |
| 'reranker_model' | '/path/to/bge-reranker-large' |
| 'top_n' | 1 |
| 'enable_rerank' | False |
| 'MRR' | 0.8 |
| 'Hit' | 0.8 |
```

### SuperBenchmark
You can run retrieval superbenchmark by the following commands.
```
cd intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/evaluation/retriever
python retrieval_superbenchmark.py \
--index_file_jsonl_path /path/to/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/evaluation/data_augmentation/candidate_context.jsonl \
--query_file_jsonl_path /path/to/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/evaluation/data_augmentation/example.jsonl \
--vector_database Chroma \
--embedding_model <embedding model name or path> \
--llm_model <llm model name or path> \
--reranker_model <reranker model name or path>
```

This will run benchmark multiple times based on the following different parameter values and output the parameter values that achieve the maximum MRR and Hit.

**Adjustable Parameters**:
- `retrieval_type`: ['default','child_parent','bm25']
- `polish`: [True, False]
- `search_type`: ['similarity','mmr','similarity_score_threshold']
- `k`: [1, 3, 5]
- `fetch_k`: [5, 10, 20]
- `score_threshold`: [0.3, 0.5, 0.7]
- `top_n`: [1, 3, 5, 10]
- `enable_rerank`: [True, False]

**Result**:
max_MRR 
```
|  Parameter & Result  | Value  |
|  :----:  | :----:  |
| 'index_file_jsonl_path'  | '/path/to/candidate_context.jsonl' |
| 'query_file_jsonl_path'  | '/path/to/example.jsonl' |
| 'vector_database'  | 'Chroma'|
| 'embedding_model' | '/path/to/bge-large-en-v1.5' |
| 'retrieval_type' | 'default' |
| 'polish' | True |
| 'search_type' | 'similarity' |
| 'llm_model' | '/path/to/neural-chat-7b-v3-1/' |
| 'k' | 1 |
| 'fetch_k' | 5 |
| 'score_threshold' | 0.3 |
| 'reranker_model' | '/path/to/bge-reranker-large' |
| 'top_n' | 1 |
| 'enable_rerank' | True |
| 'MRR' | 0.7 |
| 'Hit' | 0.7 |
```
...
max_Hit
```
|  Parameter & Result  | Value  |
|  :----:  | :----:  |
| 'index_file_jsonl_path'  | '/path/to/candidate_context.jsonl' |
| 'query_file_jsonl_path'  | '/path/to/example.jsonl' |
| 'vector_database'  | 'Chroma'|
| 'embedding_model' | '/path/to/bge-large-en-v1.5' |
| 'retrieval_type' | 'default' |
| 'polish' | True |
| 'search_type' | 'similarity' |
| 'llm_model' | '/path/to/neural-chat-7b-v3-1/' |
| 'k' | 1 |
| 'fetch_k' | 20 |
| 'score_threshold' | 0.3 |
| 'reranker_model' | '/path/to/bge-reranker-large' |
| 'top_n' | 3 |
| 'enable_rerank' | True |
| 'MRR' | 0.7 |
| 'Hit' | 0.7 |
```
...

## 3. Rag Benchmark
### Installation
Please ensure the installation of requirements for NeuralChat and retrieval plugin first by the following commands.
```
git clone https://github.com/intel/intel-extension-for-transformers.git
cd intel-extension-for-transformers/intel_extension_for_transformers/neural_chat
pip install -r requirements.txt
cd pipeline/plugins/retrieval
pip install -r requirements.txt
```
After that, please install dependency using the following commands.
```
cd intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/evaluation/framework
pip install -r requirements.txt
```

### Benchmark
You can run rag benchmark by the following commands.
```
cd intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/evaluation/framework
bash ragas_benchmark.sh \
--ground_truth_file=/path/to/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/evaluation/data_augmentation/ground_truth.jsonl \
--input_path=/path/to/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/evaluation/data_augmentation/data.txt \
--vector_database=Chroma \
--embedding_model=<embedding model name or path> \
--llm_model=<llm model name or path> \
--reranker_model=<reranker model name or path>
```

**Some Important Arguments**:
- `ground_truth_file`: The path of JSON data including question, context, and ground_truth, where each line is a dict like this:```{"question": str, "context": List[str], "ground_truth": str}```. See [ground_truth.jsonl](https://github.com/intel/intel-extension-for-transformers/blob/master/intel_extension_for_transformers/neural_chat/tools/evaluation/data_augmentation/ground_truth.jsonl) for a data file. The `"question"` of `answer_file` and `ground_truth_file` should correspond one-to-one.
- `input_path`: The path of the file/folder/link of the content.
- `use_openai_key`: Whether to utilize OpenAI for running ragas to compute the score. If you’re using openai, ensure you have your OpenAI key ready and available in your environment by `export OPENAI_API_KEY=xxx`. The default value is False.
- `vector_database`: The vector database for constructing the knowledge base.
- `embedding_model`: The name or path for the text embedding model. The default value is "BAAI/bge-base-en-v1.5". Other options are "BAAI/bge-large-en-v1.5", "thenlper/gte-large", "infgrad/stella-base-en-v2", "thenlper/gte-base", "intfloat/e5-large-v2", "hkunlp/instructor-xl", and "hkunlp/instructor-large".
- `llm_model`: The name or path for the LLM model.
- `reranker_model`: The name or path for the reranker model.
- `retrieval_type`: The type of the retriever. The default value is "default". The other options are "child_parent" and "bm25".
- `polish`: Whether to polish the input query before processing. The default value is False.
- `search_type`: Type of search to perform. The default value is "similarity". The other options are "mmr" and "similarity_score_threshold".
- `k`: The number of the returned most similar documents. The default value is 1.
- `fetch_k`: The number of Documents to fetch to pass to MMR algorithm. The default value is 5.
- `score_threshold`: The similar score threshold for the retrieved documents. The default value is 0.3.
- `top_n`: The return number of the reranker model. The default value is 1.
- `enable_rerank`: Whether to enable retrieval then rerank pipeline. The default value is False.
- `max_chuck_size`: The max token length for a single chuck in the knowledge base. The default value is 256.
- `temperature`: The value is used to modulate the next token probabilities, and will influence the distribution of similarity scores. The default value is 0.01.
- `top_k`: The number of highest probability vocabulary tokens to keep for top-k-filtering. The default value is 1.
- `top_p`: If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation. The default value is 0.1.
- `repetition_penalty`: The parameter for repetition penalty. 1.0 means no penalty. The default value is 1.0.
- `num_beams`: Number of beams for beam search. 1 means no beam search. The default value is 1.
- `do_sample`: Whether or not to use sampling; use greedy decoding otherwise. The default value is False.

**Result**:
The result will include all parameter values and values of Average Answer Relevancy, Average Faithfulness, Average Context Recall, Average Context Precision.
```
|  Parameter & Result  | Value  |
|  :----:  | :----:  |
| "ground_truth_file"  | "ground_truth.jsonl" |
| "input_path" | "data.txt" |
| "vector_database"  | "Chroma" |
| "embedding_model" | "/path/to/bge-large-en-v1.5" |
| "retrieval_type" | "default" |
| "polish" | True |
| "search_type" | "similarity" |
| "llm_model" | "/path/to/neural-chat-7b-v3-1/" |
| "k" | 1 |
| "fetch_k" | 5 |
| "score_threshold" | 0.3 |
| "reranker_model" | "/path/to/bge-reranker-large" |
| "top_n" | 1 |
| "enable_rerank" | True |
| "max_chuck_size" | 256 |
| "temperature" | 0.01 |
| "top_k" | 1 |
| "top_p" | 0.1 |
| "repetition_penalty" | 1.0 |
| "num_beams" | 1 |
| "do_sample" | True |
| "answer_relevancy_average" | 0.937748267362332 |
| "faithfulness_average" | 0.5833333333333333 |
| "context_recall_average" | 1.0 |
| "context_precision_average" | 0.49999999995 |
```

### SuperBenchmark
You can run rag superbenchmark by the following commands.
```
cd intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/evaluation/framework
python ragas_benchmark.py \
--ground_truth_file /path/to/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/evaluation/data_augmentation/ground_truth.jsonl \
--input_path /path/to/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/evaluation/data_augmentation/data.txt \
--vector_database Chroma \
--embedding_model <embedding model name or path> \
--llm_model <llm model name or path> \
--reranker_model <reranker model name or path>
```

If you utilize OpenAI for running ragas, ensure you have your OpenAI key ready and available in your environment. This will make multiple calls to the OpenAI API, please be aware of your costs.
```
cd intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/evaluation/framework
export OPENAI_API_KEY=xxx
python ragas_benchmark.py \
--ground_truth_file /path/to/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/evaluation/data_augmentation/ground_truth.jsonl \
--input_path /path/to/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/evaluation/data_augmentation/data.txt \
--use_openai_key \
--vector_database Chroma \
--embedding_model <embedding model name or path> \
--llm_model <llm model name or path> \
--reranker_model <reranker model name or path>
```

This will run benchmark multiple times based on the following different parameter values and output the parameter values that achieve the maximum Average Answer Relevancy, Average Faithfulness, Average Context Recall, Average Context Precision.

**Adjustable Parameters**:
- `retrieval_type`: ['default','child_parent','bm25']
- `polish`: [True, False]
- `search_type`: ['similarity','mmr','similarity_score_threshold']
- `k`: [1, 3, 5]
- `fetch_k`: [5, 10, 20]
- `score_threshold`: [0.3, 0.5, 0.7]
- `top_n`: [1, 3, 5, 10]
- `enable_rerank`: [True, False]
- `max_chuck_size`: [256, 512, 768, 1024]
- `temperature`: [0.01, 0.05, 0.1, 0.3, 0.5, 0.7]
- `top_k`: [1, 3, 10, 20]
- `top_p`: [0.1, 0.3, 0.5, 0.7]
- `repetition_penalty`: [1.0, 1.1, 1.3, 1.5, 1.7]
- `num_beams`: [1, 3, 10, 20]
- `do_sample`: [True, False]

**Result**:
max_answer_relevancy_average
```
|  Parameter & Result  | Value  |
|  :----:  | :----:  |
| "ground_truth_file"  | "ground_truth.jsonl" |
| "input_path" | "data.txt" |
| "vector_database"  | "Chroma" |
| "embedding_model" | "/path/to/bge-large-en-v1.5" |
| "retrieval_type" | "default" |
| "polish" | True |
| "search_type" | "similarity" |
| "llm_model" | "/path/to/neural-chat-7b-v3-1/" |
| "k" | 1 |
| "fetch_k" | 5 |
| "score_threshold" | 0.3 |
| "reranker_model" | "/path/to/bge-reranker-large" |
| "top_n" | 1 |
| "enable_rerank" | True |
| "max_chuck_size" | 256 |
| "temperature" | 0.01 |
| "top_k" | 1 |
| "top_p" | 0.1 |
| "repetition_penalty" | 1.0 |
| "num_beams" | 20 |
| "do_sample" | True |
| "answer_relevancy_average" | 0.9533325665270252 |
| "faithfulness_average" | 0.5083333333333333 |
| "context_recall_average" | 1.0 |
| "context_precision_average" | 0.49999999995 |
```
...
max_faithfulness_average
```
|  Parameter & Result  | Value  |
|  :----:  | :----:  |
| "ground_truth_file"  | "ground_truth.jsonl" |
| "input_path" | "data.txt" |
| "vector_database"  | "Chroma" |
| "embedding_model" | "/path/to/bge-large-en-v1.5" |
| "retrieval_type" | "default" |
| "polish" | True |
| "search_type" | "similarity" |
| "llm_model" | "/path/to/neural-chat-7b-v3-1/" |
| "k" | 1 |
| "fetch_k" | 5 |
| "score_threshold" | 0.3 |
| "reranker_model" | "/path/to/bge-reranker-large" |
| "top_n" | 1 |
| "enable_rerank" | True |
| "max_chuck_size" | 256 |
| "temperature" | 0.01 |
| "top_k" | 1 |
| "top_p" | 0.1 |
| "repetition_penalty" | 1.0 |
| "num_beams" | 1 |
| "do_sample" | True |
| "answer_relevancy_average" | 0.9354267206448277 |
| "faithfulness_average" | 0.675 |
| "context_recall_average" | 1.0 |
| "context_precision_average" | 0.49999999995 |
```
...
max_context_recall_average
```
|  Parameter & Result  | Value  |
|  :----:  | :----:  |
| "ground_truth_file"  | "ground_truth.jsonl" |
| "input_path" | "data.txt" |
| "vector_database"  | "Chroma" |
| "embedding_model" | "/path/to/bge-large-en-v1.5" |
| "retrieval_type" | "default" |
| "polish" | True |
| "search_type" | "similarity" |
| "llm_model" | "/path/to/neural-chat-7b-v3-1/" |
| "k" | 1 |
| "fetch_k" | 5 |
| "score_threshold" | 0.3 |
| "reranker_model" | "/path/to/bge-reranker-large" |
| "top_n" | 1 |
| "enable_rerank" | True |
| "max_chuck_size" | 256 |
| "temperature" | 0.01 |
| "top_k" | 1 |
| "top_p" | 0.1 |
| "repetition_penalty" | 1.0 |
| "num_beams" | 1 |
| "do_sample" | True |
| "answer_relevancy_average" | 0.9354267206448277 |
| "faithfulness_average" | 0.675 |
| "context_recall_average" | 1.0 |
| "context_precision_average" | 0.49999999995 |
```
...
max_context_precision_average
```
|  Parameter & Result  | Value  |
|  :----:  | :----:  |
| "ground_truth_file"  | "ground_truth.jsonl" |
| "input_path" | "data.txt" |
| "vector_database"  | "Chroma" |
| "embedding_model" | "/path/to/bge-large-en-v1.5" |
| "retrieval_type" | "default" |
| "polish" | True |
| "search_type" | "similarity" |
| "llm_model" | "/path/to/neural-chat-7b-v3-1/" |
| "k" | 1 |
| "fetch_k" | 5 |
| "score_threshold" | 0.3 |
| "reranker_model" | "/path/to/bge-reranker-large" |
| "top_n" | 1 |
| "enable_rerank" | True |
| "max_chuck_size" | 256 |
| "temperature" | 0.01 |
| "top_k" | 1 |
| "top_p" | 0.1 |
| "repetition_penalty" | 1.1 |
| "num_beams" | 1 |
| "do_sample" | True |
| "answer_relevancy_average" | 0.7429146997306499 |
| "faithfulness_average" | 0.6666666666666667 |
| "context_recall_average" | 1.0 |
| "context_precision_average" | 0.49999999995 |
```
...
