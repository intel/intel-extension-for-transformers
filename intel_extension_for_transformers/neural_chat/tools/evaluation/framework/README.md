# Ragas Evaluation

## 1. Introduction
Ragas is a framework that helps you evaluate your Retrieval Augmented Generation (RAG) pipelines. We provide a script to use Ragas based on data files.

## 2. Requirements
```
git clone https://github.com/intel/intel-extension-for-transformers.git
cd intel-extension-for-transformers/intel_extension_for_transformers/neural_chat
pip install -r requirements.txt
cd pipeline/plugins/retrieval
pip install -r requirements.txt
cd ../../../
cd tools/evaluation/framework
pip install -r requirements.txt
```

## 3. Evaluate Retrieval
```
cd intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/evaluation/framework
python ragas_evaluation.py \
--answer_file /path/to/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/evaluation/data_augmentation/answer.jsonl \
--ground_truth_file /path/to/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/evaluation/data_augmentation/ground_truth.jsonl \
--openai_api_key <your openai api key>
```

**Some Important Arguments**:
- `answer_file`: The path of JSON data including question and answer, where each line is a dict like this:```{"question": str, "answer": str}```. See [answer.jsonl](https://github.com/intel/intel-extension-for-transformers/blob/master/intel_extension_for_transformers/neural_chat/tools/evaluation/data_augmentation/answer.jsonl) for a data file.

- `ground_truth_file`: The path of JSON data including question, context, and ground_truth, where each line is a dict like this:```{"question": str, "context": List[str], "ground_truth": str}```. See [ground_truth.jsonl](https://github.com/intel/intel-extension-for-transformers/blob/master/intel_extension_for_transformers/neural_chat/tools/evaluation/data_augmentation/ground_truth.jsonl) for a data file. The `"question"` of `answer_file` and `ground_truth_file` should correspond one-to-one.

- `openai_api_key`: This guide utilizes OpenAI for running some metrics, so ensure you have your OpenAI key ready and available in your environment.

## 4. Result
The results include your input question, answer, contexts, ground_truth, as well as output answer relevancy, faithfulness, context_recall, context_precision.
```
    question     answer   contexts ground_truth  answer_relevancy  faithfulness  context_recall  context_precision
0  What t...  The or...  [We ai...  open s...     0.900788          0.500000           1.0             1.0
1  What a...  The co...  [Our w...  The co...     0.985826          0.250000           1.0             0.0
......
```
