<h1 align="center" id="title"><img align="center" src="./src/lib/assets/favicon.png" alt="project-image">Neural Chat</h1>

<h2>🚀 Demo</h2>

[http://neuralstudio.intel.com/NeuralChat](http://neuralstudio.intel.com/NeuralChat)

<h2>📸 Project Screenshots:</h2>

<img src="https://i.imgur.com/7vjfTlv.png" alt="project-screenshot" width="800" height="400/">

<img src="https://i.imgur.com/oHXlHvB.png" alt="project-screenshot" width="800" height="400/">

<img src="https://i.imgur.com/fwHWTo4.png" alt="project-screenshot" width="800" height="400/">


<h2>🧐 Features</h2>

Here're some of the project's features:

- Basic mode：Choose the best model from different domains to chat.
- Advanced mode：Select different models/different knowledge base chats, and you can customize the corresponding parameters.
- hint：Start a chat according to the prompt, which is the first sentence of the chat.
- New Topic: Clear the current context and restart the chat.
- Txt2Img: Generate an image based on the current answer, and you can hover to zoom in the image.


<h2>🛠️ Get it Running:</h2>

1. Clone the repo.

2. cd command to the current folder.

3. Modify the required .env variables.
    ```
    LLMA_URL=
    GPT_J_6B_URL=
    KNOWLEDGE_URL=
    TXT2IMG=

    ```
4. Execute `npm install` to install the corresponding dependencies.

5. Execute `npm run dev` in both enviroments