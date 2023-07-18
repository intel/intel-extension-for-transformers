Memory Controller
======
1. [Introduction](#introduction)
2. [Get Started](#get-started)
3. [Example](#example)

## Introduction

The conversation agent treats each incoming user query as an isolated instance by default, triggering the language model to generate a suitable response. This approach is effective when the available knowledge base covers the topic of the user query. However, user queries can be sophisticated, and the general knowledge base may not provide adequate answers. In such cases, it is essential to introduce a memory component to collect user feedback during the conversation and improve future responses.

## Get Started
The conversation memory represents the information within a conversation session between the user and the language model. We consider two types of memories during the conversation: the `chat memory` and the `entity memory`. The chat memory refers to the interaction context, and it is common for users to refer to previously mentioned terms or information using pronouns. This can lead to confusion when there is no conversation memory. Additionally, users may provide feedback to correct any false cognitions during the conversation. The stored chat memory is helpful for rectifying inaccuracies and improving future responses. To address situations where conversations contain irrelevant information, we introduce the entity memory as an alternative. The entity memory captures entity-related descriptions in the conversation, using the `spacy` library to identify entity names and save the memory in the {name, context} format. The subsequent response generation process includes the entity information to achieve better results.

We aim to make it easy for users to include the memory component. Users can activate the conversation chain directly to test its performance.
```bash
python chat_with_memory.py --model_path XXX
```

Here, `--model_path` is the model id or the local LLM file to triger the interaction. Users have the flexibility to adjust the generation parameters, such as `--temperature`, `--penalty`, and `--max_length`, in order to optimize the generation performance for different application scenarios.

Regarding the memory types, we offer three options: `ConversationBufferMemory`, `ConversationBufferWindowMemory`, and `SpacyEntityMemory`. The choice of memory type depends on the model's inherent capabilities and suitability. Considering that conversation memory adds more information to the input prompt, we recommend using ConversationBufferWindowMemory when dealing with models of around 7B-weight level. Having an excessively long chat history can pose challenges for the language model in understanding the input prompts, resulting in unsatisfactory generation results. Therefore, limiting the length of the conversation history is advisable.

## Example
Below is an example about the actual performance of the introduced memory component.

Without memory:
```
AI: SAL is currently working on a project to develop an AI-based system to detect and diagnose diseases.​

Human: Nope. the Systems Architecture Lab (SAL) worked on new instructions to improve the efficiency of submitting work to and synchronizing work among the compute cores and the dedicated accelerators.​

AI: Ah, I see. SAL also worked on optimizing the communication between the cores and the accelerators to ensure that the data transfer is fast and efficient.​

Human: What is SAL working on?​

AI: SAL is currently working on a project to develop an AI-based system to detect and diagnose diseases.
```

With memory:
```
Human: What is SAL working on?​

AI: SAL is currently working on a project to develop an AI-based system to detect and diagnose diseases.​

Human: Nope. the Systems Architecture Lab (SAL) worked on new instructions to improve the efficiency of submitting work to and synchronizing work among the compute cores and the dedicated accelerators.​

AI: Oh, okay. I see. Thanks for the clarification.​

Human: What is SAL working on?​

AI: SAL working on new instructions to improve the efficiency of submitting work to and synchronizing work among the compute cores and the dedicated accelerators.
```

In the above example, it is evident that incorporating memory can help mitigate the model's tendency to fabricate nonexistent information about the target entity. By utilizing user feedback, the generation results can be optimized during subsequent conversations.
