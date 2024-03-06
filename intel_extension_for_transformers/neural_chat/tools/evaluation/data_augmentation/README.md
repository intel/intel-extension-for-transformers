# Retrieval Data Augmentation

## 1. Introduction
In this example, we show how to do data augmentation to construct a retrieval dataset. Specifically, the effect is to generate specific open-ended questions based on the context of the input file provided. The questions are directly related to the context to form a query-positive pair, suitable for use in constructing a retrieval dataset.

## 2. Requirements
```
git clone https://github.com/intel/intel-extension-for-transformers.git
cd intel-extension-for-transformers/intel_extension_for_transformers/neural_chat
pip install -r requirements.txt
cd pipeline/plugins/retrieval
pip install -r requirements.txt
```

* **On CPU**
```
cd intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/evaluation/data_augmentation
pip install -r requirements_cpu.txt
```

* **On CUDA**
```
cd intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/evaluation/data_augmentation
pip install -r requirements_cuda.txt
```

## 3. Retrieval Dataset Construction
* **On CPU**
```
cd intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/evaluation
python -m data_augmentation.retrieval_dataset_construction \
--llm_model <llm model path> \
--embedding_model <embedding model path> \
--input <your input file path>
```

* **On CUDA**
```
cd intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/evaluation
python -m data_augmentation.retrieval_dataset_construction \
--llm_model <llm model path> \
--embedding_model <embedding model path> \
--input <your input file path> \
--use_gpu_for_searching True
```

**Some Important Arguments**:
- `llm_model`: The path for the LLM model.
- `embedding_model`: The path for the text embedding model.
- `input`: The path of the file/folder/link of the content.
- `output`: The name of output files. The default value is 'data'. The default output files are 'data.jsonl', 'data_minedHN.jsonl', 'data_minedHN_split.jsonl'.
- `temperature`: The value is used to modulate the next token probabilities, and will influence the distribution of similarity scores. The default value is 0.8.
- `top_p`: If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation. The default value is 0.9.
- `top_k`: The number of highest probability vocabulary tokens to keep for top-k-filtering. The default value is 40.
- `repetition_penalty`: The parameter for repetition penalty. 1.0 means no penalty. The default value is 2.0.
- `max_new_tokens`: The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt. The default value is 48.
- `do_sample`: Whether or not to use sampling ; use greedy decoding otherwise. The default value is True.
- `num_beams`: Number of beams for beam search. 1 means no beam search. The default value is 2.
- `num_return_sequences`: The number of independently computed returned sequences for each element in the batch. The default value is 2.
- `use_cache`: Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding. The default value is True.
- `range_for_sampling`: The range to sample negatives. For example, `2-100` means sampling `negative_number` negatives from top2-top200 documents. You can set a larger value to reduce the difficulty of negatives (e.g., set it `60-300` to sample negatives from top60-300 passages). The default value is '2-10'.
- `negative_number`: The number of sampled negatives. The default value is 5.
- `use_gpu_for_searching`: Whether to use faiss-gpu to retrieve negatives. The default value is False.
- `similarity_threshold`: The cosine similarity threshold used to filter the generated queries. The default value is 0.6.

## 4. Result
Three files will be generated. The default output files are `data.jsonl`, `data_minedHN.jsonl`, `data_minedHN_split.jsonl`. The third is the final output dataset, where each line is a dict like this:
```
{"query": str, "pos": List[str], "neg": List[str]}
```
`query` is the query, and `pos` is a positive text, based on the context of the input file provided, `neg` is a list of negative texts.
See [augmented_example.jsonl](https://github.com/intel/intel-extension-for-transformers/blob/master/intel_extension_for_transformers/neural_chat/tools/embedding_finetune/augmented_example.jsonl) for a data file.
