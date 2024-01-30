NeuralChat Notebooks
===========================

Welcome to use Jupyter Notebooks to explore how to build and customize chatbots across a wide range of platforms, including Intel Xeon CPU(ICX and SPR), Intel XPU, Intel Habana Gaudi1/Gaudi2, and Nvidia GPU. Dive into our detailed guide to discover how to develop chatbots on these various computing platforms.

| Chapter | Section                                       | Description                                                | Notebook Link                                           |
| ------- | --------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------- |
| 1       | Building Chatbots                             |                                                           |                                                         |
| 1.1     | Building Chatbot on Intel CPU ICX             | Learn how to create chatbot on ICX                      | [Notebook](./notebooks/build_chatbot_on_icx.ipynb) |
| 1.2     | Building Chatbot on Intel CPU SPR             | Learn how to create chatbot on SPR                      | [Notebook](./notebooks/build_chatbot_on_spr.ipynb) |
| 1.3     | Building Chatbot on Intel XPU                 | Learn how to create chatbot on XPU                      | [Notebook](./notebooks/build_chatbot_on_xpu.ipynb) |
| 1.4     | Building Chatbot on Habana Gaudi1/Gaudi2      | Learn how to create chatbot on Habana Gaudi1/Gaudi2     | [Notebook](./notebooks/build_chatbot_on_habana_gaudi.ipynb) |
| 1.5     | Building Chatbot on Nvidia A100               | Learn how to create chatbot on Nvidia A100              | [Notebook](./notebooks/build_chatbot_on_nv_a100.ipynb)   |
| 1.6     | Building Chatbot on Intel CPU Windows PC      | Learn how to create chatbot on Windows PC               | [Notebook](./notebooks/build_talkingbot_on_pc.ipynb) |
| 2       | Deploying Chatbots                            |                                                           |                                                         |
| 2.1     | Deploying Chatbot on Intel CPU ICX            | Learn how to deploy chatbot on ICX                    | [Notebook](./notebooks/deploy_chatbot_on_icx.ipynb) |
| 2.2     | Deploying Chatbot on Intel CPU SPR            | Learn how to deploy chatbot on SPR                    | [Notebook](./notebooks/deploy_chatbot_on_spr.ipynb) |
| 2.3     | Deploying Chatbot on Intel XPU                | Learn how to deploy chatbot on Intel XPU              | [Notebook](./notebooks/deploy_chatbot_on_xpu.ipynb) |
| 2.4     | Deploying Chatbot on Habana Gaudi1/Gaudi2     | Learn how to deploy chatbot on Habana Gaudi1/Gaudi2   | [Notebook](./notebooks/deploy_chatbot_on_habana_gaudi.ipynb) |
| 2.5     | Deploying Chatbot on Nvidia A100              | Learn how to deploy chatbot on A100                   | [Notebook](./notebooks/deploy_chatbot_on_nv_a100.ipynb) |
| 2.6     | Deploying Chatbot with Load Balance           | Learn how to deploy chatbot with load balance         | [Notebook](./notebooks/chatbot_with_load_balance.ipynb) |
| 2.7     | Deploying End-to-end text Chatbot on Intel CPU SPR  | Learn how to deploy an end to end text chatbot on Intel CPU SPR including frontend GUI and backend | [Notebook](./notebooks/setup_text_chatbot_service_on_spr.ipynb) |
| 2.8     | Deploying End-to-end talkingbot on Intel CPU SPR  | Learn how to deploy an end to end talkingbot on Intel CPU SPR including frontend GUI and backend | [Notebook](./notebooks/setup_talking_chatbot_service_on_spr.ipynb) |
| 2.9     | Deploying End-to-end text Chatbot witch caching on Intel CPU SPR  | Learn how to deploy an end to end text chatbot with plugin on Intel CPU SPR including frontend GUI and backend | [Notebook](./notebooks/setup_text_chatbot_with_caching_on_spr.ipynb) |
| 3       | Optimizing Chatbots                         |                                                            |                                                         |
| 3.1     | Enabling Chatbot with BF16 Optimization on SPR        | Learn how to optimize chatbot using mixed precision on SPR | [Notebook](./notebooks/amp_optimization_on_spr.ipynb) |
| 3.2     | Enabling Chatbot with BF16 Optimization on Habana Gaudi1/Gaudi2 | Learn how to optimize chatbot using mixed precision on Habana Gaudi1/Gaudi2 | [Notebook](./notebooks/amp_optimization_on_habana_gaudi.ipynb) |
| 3.3     | Enabling Chatbot with BitsAndBytes Optimization on Nvidia A100 | Learn how to optimize chatbot using BitsAndBytes on Nvidia A100 | [Notebook](./notebooks/bits_and_bytes_optimization_on_nv_a100.ipynb) |
| 3.4     | Enabling Chatbot with Weight Only INT4 Optimization on SPR | Learn how to optimize chatbot using ITREX LLM graph Weight Only INT4 on SPR | [Notebook](./notebooks/itrex_llm_graph_int4_optimization_on_spr.ipynb) |
| 4       | Fine-Tuning Chatbots                           |                                                            |                                                         |
| 4.1     | Fine-tuning on SPR (Single Node)               | Learn how to fine-tune chatbot on SPR with single node | [Notebook](./notebooks/single_node_finetuning_on_spr.ipynb) |
| 4.2     | Fine-tuning on SPR (Multiple Nodes)            | Learn how to fine-tune chatbot on SPR with multiple nodes | [Notebook](./notebooks/multi_node_finetuning_on_spr.ipynb) |
| 4.3     | Fine-tuning on Habana Gaudi1/Gaudi2 (Single Card) | Learn how to fine-tune on Habana Gaudi1/Gaudi2 with single card | [Notebook](./notebooks/single_card_finetuning_on_habana_gaudi.ipynb) |
| 4.4     | Fine-tuning on Nvidia A100 (Single Card)       | Learn how to fine-tune chatbot on Nvidia A100 | [Notebook](./notebooks/finetuning_on_nv_a100.ipynb) |
| 4.5     | Finetune Neuralchat on NVIDIA GPU       | Learn how to fine-tune Neuralchat on Nvidia GPU | [Notebook](./notebooks/finetune_neuralchat_v2_on_Nvidia_GPU.ipynb) |
| 4.6     | Finetuning or RAG for external knowledge       | Learn how to fine-tune or RAG for external knowledge | [Notebook](./notebooks/Finetuning_or_RAG_for_external_knowledge.ipynb) |
| 5       | Customizing Chatbots                          |                                                          |                                                         |
| 5.1     | Enabling Plugins to Customize Chatbot         | Learn how to customize chatbot using plugins             | [Notebook](./notebooks/customize_chatbot_with_plugins.ipynb) |
| 5.2     | Enabling Fine-tuned Models in Chatbot         | Learn how to customize chatbot using fine-tuned models   | [Notebook](./notebooks/customize_chatbot_with_finetuned_models.ipynb) |
| 5.3     | Enabling Optimized Models in Chatbot          | Learn how to customize chatbot using optimized models    | [Notebook](./notebooks/customize_chatbot_with_optimized_models.ipynb) |
| 5.4     | Enabling New LLM Models to Customize Chatbot  | Learn how to use new LLM models to customize chatbot     | [Notebook](./notebooks/customize_chatbot_with_new_llm_models.ipynb) |
| 6       | Extension API                           |                                                          |                                                         |
| 6.1     | Transformers Extension API         | Learn how to use transformer extension API             | [Notebook](./notebooks/transformers_extension_api.ipynb) |
| 6.2     | Langchain Extension API         | Learn how to use langchain extension API   | [Notebook](./notebooks/langchain_extension_api.ipynb) |
