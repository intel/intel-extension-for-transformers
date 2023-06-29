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

### Service

Under construction.


To simplify the deployment process, we have also included Docker files for each part, allowing for easy and efficient building of the whole workflow service. These Docker files provide a standardized environment and streamline the deployment process, ensuring smooth execution of the chatbot service.


# Purpose of the NeuralChat for Intel Architecture

- Demonstrate the AI workloads and deep learning models Intel has optimized and validated to run on Intel hardware

- Show how to efficiently execute, train, and deploy Intel-optimized models

- Make it easy to get started running Intel-optimized models on Intel hardware in the cloud or on bare metal

DISCLAIMER: These scripts are not intended for benchmarking Intel platforms. For any performance and/or benchmarking information on specific Intel platforms, visit https://www.intel.ai/blog.

Intel is committed to the respect of human rights and avoiding complicity in human rights abuses, a policy reflected in the Intel Global Human Rights Principles. Accordingly, by accessing the Intel material on this platform you agree that you will not use the material in a product or application that causes or contributes to a violation of an internationally recognized human right.

## Models

To the extent that any model(s) are referenced by Intel or accessed using tools or code on this site those models are provided by the third party indicated as the source. Intel does not create the model(s) and does not warrant their accuracy or quality. You understand that you are responsible for understanding the terms of use and that your use complies with the applicable license.

## Datasets

To the extent that any public or datasets are referenced by Intel or accessed using tools or code on this site those items are provided by the third party indicated as the source of the data. Intel does not create the data, or datasets, and does not warrant their accuracy or quality. By accessing the public dataset(s) you agree to the terms associated with those datasets and that your use complies with the applicable license. 
<br>
[Alpaca](https://github.com/tatsu-lab/stanford_alpaca)

Intel expressly disclaims the accuracy, adequacy, or completeness of any public datasets, and is not liable for any errors, omissions, or defects in the data, or for any reliance on the data. Intel is not liable for any liability or damages relating to your use of public datasets.
