Welcome to the Text Chatbot! This example introduces how to deploy the Text Chatbot system and guides you through setting up both the backend and frontend components. You can deploy this text chatbot on various platforms, including Intel XEON Scalable Processors, Habana's Gaudi processors (HPU), Intel Data Center GPU and Client GPU, Nvidia Data Center GPU and Client GPU.

| Section                         |                     Link                              |
| --------------------------------| ------------------------------------------------------|
| Backend Setup                   | [Backend README](./backend/xeon/README.md)            |
| Frontend Setup                  | [Frontend README](../../../ui/gradio/basic/README.md) |

You can enhance the capabilities of the Text Chatbot by enabling plugins, such as the cache plugin. This plugin is designed to help you reduce costs effectively by caching query results, resulting in fewer requests and tokens sent to the Language Model service. As a result, it offers superior query throughput compared to standard Language Model services. To deploy a Text Chatbot with caching functionality, please refer to the instructions provided in the [README](./backend_with_cache/README.md) for backend setup.
