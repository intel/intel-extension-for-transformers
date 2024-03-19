# Ragas Evaluation

## 1. Introduction
[Ragas](https://github.com/explodinggradients/ragas) is a framework that helps you evaluate your Retrieval Augmented Generation (RAG) pipelines. We provide a script to use Ragas based on data files. We use four metrics: answer relevancy, faithfulness, context recall, context precision.
* **Answer relevancy** focuses on assessing how pertinent the generated answer is to the given prompt. A lower score is assigned to answers that are incomplete or contain redundant information and higher scores indicate better relevancy.
* **Faithfulness** measures the factual consistency of the generated answer against the given context. It is calculated from answer and retrieved context. The answer is scaled to (0,1) range. Higher is the better.
* **Context recall** measures the extent to which the retrieved context aligns with the annotated answer, treated as the ground truth. It is computed based on the ground truth and the retrieved context, and the values range between 0 and 1, with higher values indicating better performance.
* **Context precision** is a metric that evaluates whether all of the ground-truth relevant items present in the contexts are ranked higher or not. Ideally all the relevant chunks must appear at the top ranks. This metric is computed using the question, ground_truth and the contexts, with values ranging between 0 and 1, where higher scores indicate better precision.

## 2. Installation
Please install dependency using the following commands.
```
git clone https://github.com/intel/intel-extension-for-transformers.git
cd intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/evaluation/framework
pip install -r requirements.txt
```

## 3. Evaluate RAG
* **OpenAI**
By default, ragas use OpenAI’s API to compute the score. If you’re using this metric, ensure that you set the environment key OPENAI_API_KEY with your API key.
```
cd intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/evaluation/framework
python ragas_evaluation.py \
--answer_file /path/to/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/evaluation/data_augmentation/answer.jsonl \
--ground_truth_file /path/to/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/evaluation/data_augmentation/ground_truth.jsonl \
--openai_api_key <your openai api key>
```
* **Langchain**
You can also try other LLMs for evaluation using Langchain.
```
cd intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/evaluation/framework
python ragas_evaluation.py \
--answer_file /path/to/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/evaluation/data_augmentation/answer.jsonl \
--ground_truth_file /path/to/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/evaluation/data_augmentation/ground_truth.jsonl \
--llm_model <llm model path> \
--embedding_model <embedding model path>
```

**Some Important Arguments**:
- `answer_file`: The path of JSON data including question and answer, where each line is a dict like this:```{"question": str, "answer": str}```. See [answer.jsonl](https://github.com/intel/intel-extension-for-transformers/blob/master/intel_extension_for_transformers/neural_chat/tools/evaluation/data_augmentation/answer.jsonl) for a data file.
- `ground_truth_file`: The path of JSON data including question, context, and ground_truth, where each line is a dict like this:```{"question": str, "context": List[str], "ground_truth": str}```. See [ground_truth.jsonl](https://github.com/intel/intel-extension-for-transformers/blob/master/intel_extension_for_transformers/neural_chat/tools/evaluation/data_augmentation/ground_truth.jsonl) for a data file. The `"question"` of `answer_file` and `ground_truth_file` should correspond one-to-one.
- `openai_api_key`: If you utilize OpenAI for running ragas, ensure you have your OpenAI key ready and available in your environment.
- `llm_model`: If you utilize Langchain for running ragas, you should input the path for the LLM model.
- `embedding_model`: If you utilize Langchain for running ragas, you should input the path for the text embedding model. You can use "BAAI/bge-base-en-v1.5", "BAAI/bge-large-en-v1.5", "thenlper/gte-large", "infgrad/stella-base-en-v2", "thenlper/gte-base", "intfloat/e5-large-v2", "hkunlp/instructor-xl", and "hkunlp/instructor-large".

## 4. Result
The results include your input question, answer, contexts, ground_truth, as well as output answer relevancy, faithfulness, context recall, context precision.
```
    question     answer   contexts ground_truth  answer_relevancy  faithfulness  context_recall  context_precision
0  What t...  The or...  [We ai...  open s...     0.900788          0.500000           1.0             1.0
1  What a...  The co...  [Our w...  The co...     0.985826          0.250000           1.0             0.0
......
```
