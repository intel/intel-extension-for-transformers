# Finetune Embedding Model on Task-Specific Datasets

## 1. Introduction
In this example, we show how to construct the training data for finetuning the embedding model and finetuning the specific embedding model.

## 2. Requirements
* **On CPU**
```
pip install -r requirements_cpu.txt
```

* **On CUDA**
```
pip install -r requirements_cuda.txt
```
 

## 3. Training Data Construction
Train data should be a JSON file, where each line is a dict like this:

```
{"query": str, "pos": List[str], "neg": List[str]}
```

`query` is the query, and `pos` is a positive text, `neg` is a list of negative texts.
See [augmented_example.jsonl](https://github.com/intel/intel-extension-for-transformers/blob/master/intel_extension_for_transformers/neural_chat/tools/embedding_finetune/augmented_example.jsonl) for a data file.

If you have no negative texts for a query, You can use [this script](https://github.com/intel/intel-extension-for-transformers/blob/master/intel_extension_for_transformers/neural_chat/tools/embedding_finetune/mine_hard_neg.py) as follows to mine a given number of hard negatives.

### Mine Hard Negatives

Hard negatives mining is a widely used method to improve the quality of sentence embedding. 
You can mine hard negatives following this command:
* **On CPU**
```bash
python mine_hard_neg.py \
--model_name_or_path BAAI/bge-base-en-v1.5 \
--input_file example.jsonl \
--output_file augmented_example.jsonl \
--range_for_sampling 2-10 \
--negative_number 5
```
* **On CUDA**
```bash
python mine_hard_neg.py \
--model_name_or_path BAAI/bge-base-en-v1.5 \
--input_file example.jsonl \
--output_file augmented_example.jsonl \
--range_for_sampling 2-10 \
--negative_number 5 \
--use_gpu_for_searching 
```

**Some Important Arguments**:
- `input_file`: JSON data for finetuning. This script will retrieve top-k documents for each query, 
and random sample negatives from the top-k documents (not including the positive documents).
- `output_file`: path to save JSON data with mined hard negatives for finetuning
- `negative_number`: the number of sampled negatives 
- `range_for_sampling`: where to sample negative. For example, `2-100` means sampling `negative_number` negatives from top2-top200 documents. You can set a larger value to reduce the difficulty of negatives (e.g., set it `60-300` to sample negatives from top60-300 passages)
- `use_gpu_for_searching`: whether to use faiss-gpu to retrieve negatives.


## 4. Training Example
```
python finetune.py \
--output_dir BAAI/bge-base-en-v1.5_finetuned \
--model_name_or_path BAAI/bge-base-en-v1.5 \
--train_data augmented_example.jsonl \
--learning_rate 1e-5 \
--num_train_epochs 5 \
--per_device_train_batch_size 1 \
--dataloader_drop_last True \
--normalized True \
--temperature 0.02 \
--query_max_len 64 \
--passage_max_len 256 \
--train_group_size 2 \
--logging_steps 10 \
--query_instruction_for_retrieval "" \
--bf16 True
```

**Some Important Arguments**:
- `per_device_train_batch_size`: batch size in training. In most cases, a larger batch size will bring stronger performance. 
- `train_group_size`: the number of positives and negatives for a query in training.
There is always one positive, so this argument will control the number of negatives (#negatives=train_group_size-1).
Noted that the number of negatives should not be larger than the number of negatives in data `"neg": List[str]`.
Besides the negatives in this group, the in-batch negatives also will be used in fine-tuning.
- `negatives_cross_device`: share the negatives across all GPUs. This argument will extend the number of negatives.
- `learning_rate`: select an appropriate for your model. Recommend 1e-5/2e-5/3e-5 for large/base/small-scale. 
- `temperature`: It will influence the distribution of similarity scores.
- `query_max_len`: max length for query. Please set it according to the average length of queries in your data.
- `passage_max_len`: max length for passage. Please set it according to the average length of passages in your data.
- `query_instruction_for_retrieval`: instruction for query, which will be added to each query. You also can set it `""` to add nothing to the query.
- `use_inbatch_neg`: use passages in the same batch as negatives. The default value is True. 

For more training arguments please refer to [transformers.TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)


## 5. Evaluation
We provide [a simple script](https://github.com/intel/intel-extension-for-transformers/blob/master/intel_extension_for_transformers/neural_chat/tools/embedding_finetune/evaluate.py) to evaluate the model's performance. We use two metrics: MRR (Mean reciprocal rank) and Hit (Hit Ratio). MRR is an internationally accepted mechanism for evaluating search algorithms. MRR emphasizes the position of ground truth in the retrieval list, the higher it is, the better. Hit emphasizes the accuracy of retrieval, that is, whether the ground truth is included in the retrieval items.

* **Before Finetune**
```bash
python evaluate.py \
--model_name BAAI/bge-base-en-v1.5 \
--index_file_jsonl_path candidate_context.jsonl \
--query_file_jsonl_path example.jsonl
```
* **After Finetune**
```bash
python evaluate.py \
--model_name BAAI/bge-base-en-v1.5_finetuned \
--index_file_jsonl_path candidate_context.jsonl \
--query_file_jsonl_path example.jsonl
```
**Some Important Arguments:**
- `index_file_jsonl_path`: path of JSON data including candidate context where each line is a dict like this:```{"context": List[str]}```.
- `query_file_jsonl_path`: path of JSON data including queries and positives where each line is a dict like this:```{"query": str, "pos": List[str]}```.

We conducted a finetuning on an internal business dataset. The results were as follows:
* **Before Finetune**
```python
{'MRR@1': 0.7385, 'Hit@1': 0.7336}
```
* **After Finetune**
```python
{'MRR@1': 0.8297, 'Hit@1': 0.8275}
```
## 6. Verified Models
|  Model Name   | Enable  |
|  :----:  | :----:  |
| [bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)  | ✔ |
| [bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5)  | ✔ |
| [gte-large](https://huggingface.co/thenlper/gte-large)  | ✔ |
| [gte-base](https://huggingface.co/thenlper/gte-base)  | ✔ |
| [stella-base-en-v2](https://huggingface.co/infgrad/stella-base-en-v2)  | ✔ |
| [e5-large-v2](https://huggingface.co/intfloat/e5-large-v2)  | ✔ |
| [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)  | ✔ |
