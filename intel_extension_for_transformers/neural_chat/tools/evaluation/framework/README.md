# Ragas Evaluation

## 1. Introduction
Ragas is a framework that helps you evaluate your Retrieval Augmented Generation (RAG) pipelines. We provide a script to use Ragas based on data files. We use four metrics: answer relevancy, faithfulness, context recall, context precision
* **Answer relevancy**
Answer Relevancy focuses on assessing how pertinent the generated answer is to the given prompt. A lower score is assigned to answers that are incomplete or contain redundant information and higher scores indicate better relevancy.
* **Faithfulness**
Faithfulness measures the factual consistency of the generated answer against the given context. It is calculated from answer and retrieved context. The answer is scaled to (0,1) range. Higher the better.
* **Context recall**
Context recall measures the extent to which the retrieved context aligns with the annotated answer, treated as the ground truth. It is computed based on the ground truth and the retrieved context, and the values range between 0 and 1, with higher values indicating better performance.
* **Context precision**
Context Precision is a metric that evaluates whether all of the ground-truth relevant items present in the contexts are ranked higher or not. Ideally all the relevant chunks must appear at the top ranks. This metric is computed using the question, ground_truth and the contexts, with values ranging between 0 and 1, where higher scores indicate better precision.

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

## 3. Evaluate RAG
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
The results include your input question, answer, contexts, ground_truth, as well as output answer relevancy, faithfulness, context recall, context precision.
```
    question     answer   contexts ground_truth  answer_relevancy  faithfulness  context_recall  context_precision
0  What t...  The or...  [We ai...  open s...     0.900788          0.500000           1.0             1.0
1  What a...  The co...  [Our w...  The co...     0.985826          0.250000           1.0             0.0
......
```
