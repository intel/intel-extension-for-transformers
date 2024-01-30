Intel Neural Chat Dockerfile

# Fine-tuning

We provide a comprehensive pipeline on fine-tuning a customized model in Docker using [Dockerfile](./Dockerfile). It covers the process of generating custom instruction datasets, instruction templates, fine-tuning the model with these datasets, and leveraging an RLHF (Reinforcement Learning from Human Feedback) pipeline for efficient LLM fine-tuning. For detailed information and step-by-step instructions, please consult this [README file](./finetuning/README.md).


# Inference

We provide multiple plugins to augment the chatbot on top of LLM inference in Docker using [Dockerfile](./Dockerfile). Our plugins support knowledge retrieval, query caching, prompt optimization, safety checker, etc. Knowledge retrieval consists of document indexing for efficient retrieval of relevant information, including Dense Indexing based on LangChain and Sparse Indexing based on fastRAG, document rankers to prioritize the most relevant responses. Query caching enables the fast path to get the response without LLM inference and therefore improves the chat response time. Prompt optimization supports auto prompt engineering to improve user prompts, instruction optimization to enhance the model's performance, and memory controller for efficient memory utilization. For more information on these optimization techniques, please refer to this [README file](./inference/README.md).
