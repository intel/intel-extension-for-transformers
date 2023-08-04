NeuralChat
============

NeuralChat is a powerful and versatile chatbot designed to facilitate textual conversations. By providing NeuralChat with textual instructions, users can receive accurate and relevant textual responses. We provide a comprehensive workflow for building a highly customizable end-to-end chatbot service, covering model pre-training, model fine-tuning, model compression, prompt engineering, knowledge base retrieval and quick deployment.



## Fine-tuning Pipeline

We provide a comprehensive pipeline on fine-tuning a customized model. It covers the process of [generating custom instruction datasets](./fine_tuning/instruction_generator/), [instruction templates](./fine_tuning/instruction_template), [fine-tuning the model with these datasets](./fine_tuning/instruction_tuning_pipeline/), and leveraging an [RLHF (Reinforcement Learning from Human Feedback) pipeline](./fine_tuning/rlhf_learning_pipeline/) for efficient fine-tuning of the pretrained large language model (LLM). For detailed information and step-by-step instructions, please consult this [README file](./fine_tuning/README.md).


## Inference Pipeline

We focuse on optimizing the inference process of the fine-tuned customized model. It includes [auto prompt engineering](./inference/auto_prompt/) techniques for improving user prompts, [document indexing](./inference/document_indexing/README.md) for efficient retrieval of relevant information, including Dense Indexing based on [LangChain](https://github.com/hwchase17/langchain) and Sparse Indexing based on [fastRAG](https://github.com/IntelLabs/fastRAG), [document rankers](./inference/document_ranker/) to prioritize the most relevant responses, [instruction optimization](./inference/instruction_optimization/) to enhance the model's performance, and a [memory controller](./inference/memory_controller/) for efficient memory utilization. For more information on these optimization techniques, please refer to this [README file](./inference/README.md).

## Deployment

### Demo

We offer a rich demonstration of the capabilities of NeuralChat. It showcases a variety of components, including a basic frontend, an advanced frontend with enhanced features, a Command-Line interface for convenient interaction, and different backends to suit diverse requirements. For more detailed information and instructions, please refer to the [README file](./demo/README.md).

### Getting Started
#### Prepare
```bash
## Prepare Scripts
git clone https://github.com/intel/intel-extension-for-transformers.git
cd intel-extension-for-transformers/workflows/chatbot
## Prepare Data
wget https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json
## Prepare Model
git clone https://huggingface.co/mosaicml/mpt-7b-chat
## Install Dependencies
pip install langchain chromadb PyPDF2 farm-haystack==1.14.0
```

#### Indexing
```python
from inference.document_indexing.doc_index import d_load_jsonl_file, persist_embedding
documents = d_load_jsonl_file("/path/alpaca_data.json", process=False)
persist_embedding(documents, "./output", model_path="path/mpt-7b")
```

#### Inference
```python
from transformers import set_seed
from inference.generate import create_prompts, load_model, predict_stream
set_seed(1234)
instructions = "Transform the following sentence into one that shows contrast. The tree is rotten."
prompts = create_prompts([{"instruction": instruction, "input": ""} for instruction in instructions])
load_model("/path/mpt-7b-chat", "EleutherAI/gpt-neox-20b", "cpu", use_deepspeed=False)
start_time = time.time()
print("Warmup, Response: ")
for new_text in predict_stream(model_name="./mpt-7b-chat", device="cpu", prompt="Tell me about Intel Xeon.", temperature=0.1, top_p=0.75, top_k=40, repetition_penalty=1.1, num_beams=0, max_new_tokens=128, do_sample=True, use_hpu_graphs=False, use_cache=True, num_return_sequences=1):
    print(new_text, end="", flush=True)
print(f"duration: {time.time() - start_time}")
for idx, tp in enumerate(zip(prompts, instructions)):
    prompt, instruction = tp
    idxs = f"{idx+1}"
    print("=" * 30 + idxs + "=" * 30)
    print(f"Instruction: {instruction}")
    start_time = time.time()
    print("Response: ")
    first_token = True
    token_len = 0
    for new_text in predict_stream(model_name="./mpt-7b-chat", device="cpu", prompt="Tell me about Intel Xeon.", temperature=0.1, top_p=0.75, top_k=40, repetition_penalty=1.1, num_beams=0, max_new_tokens=128, do_sample=True, use_hpu_graphs=False, use_cache=True, num_return_sequences=1):
        if first_token:
            first_time_stamp = time.time()
            print(f"first token latency: {first_time_stamp - start_time}")
            first_token = False
        print(new_text, end="", flush=True)
        token_len = token_len + 1
    duration = time.time() - first_time_stamp
    print(f"duration: {time.time() - start_time}, msecond_per_token = {duration*1000/(token_len-1)}")
    print("=" * (60 + len(idxs)))
for idx, tp in enumerate(zip(prompts, instructions)):
    prompt, instruction = tp
    idxs = f"{idx+1}"
    print("=" * 30 + idxs + "=" * 30)
    print(f"Instruction: {instruction}")
    start_time = time.time()
    print("Response: ")
    out = predict(model_name="./mpt-7b-chat", device="cpu", prompt="Tell me about Intel Xeon.", temperature=0.1, top_p=0.75, top_k=40, repetition_penalty=1.1, num_beams=0, max_new_tokens=128, do_sample=True, use_hpu_graphs=False, use_cache=True, num_return_sequences=1) 
    print(f"whole sentence out = {out}")
    print(f"duration: {time.time() - start_time}")
    print("=" * (60 + len(idxs)))
```
### Disclaimer

Please refer to [DISCLAIMER](./DISCLAIMER) for details. 

The WODKFLOW SCRIPTS are not intended for benchmarking Intel platforms. For any performance and/or benchmarking information on specific Intel platforms, visit https://www.intel.ai/blog.

Intel is committed to the respect of human rights and avoiding complicity in human rights abuses, a policy reflected in the Intel Global Human Rights Principles. Accordingly, by accessing the Intel material on this platform you agree that you will not use the material in a product or application that causes or contributes to a violation of an internationally recognized human right. 