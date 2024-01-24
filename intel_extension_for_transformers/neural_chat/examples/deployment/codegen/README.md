Code generation represents another significant application of Language Model (LM) technology. NeuralChat supports various popular code generation models across different devices and provides services similar to GitHub Copilot. NeuralChat copilot is a hybrid copilot which involves real-time code generation using client PC combines with deeper server-based insight. Users have the flexibility to deploy a robust Large Language Model (LLM) in the public cloud or on-premises servers, facilitating the generation of extensive code excerpts based on user commands or comments. Additionally, users can employ an optimized LLM on their local PC as an AI assistant capable of addressing queries related to user code, elucidating code segments, refactoring, identifying and rectifying code anomalies, generating unit tests, and more.

On the server side, our platform supports deployment on a variety of hardware, including Intel XEON Scalable Processors, Habana's Gaudi processors (HPU), Intel Data Center GPU and Nvidia Data Center GPU. On the client side, we prioritize optimized models to enhance inference speed. For peak performance on client PCs, we recommend a minimum of 16GB memory.

| Section              | Link                                                              |
| ---------------------| ------------------------------------------------------------------|
| Gaudi Setup          | [Gaudi Backend](./backend/gaudi/README.md)                                |
| Xeon Setup           | [Xeon Backend](./backend/xeon/README.md)                                  |
| Client PC Setup      | [PC Backend](./backend/pc/README.md)                                      |
