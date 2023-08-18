<h1 align="center" id="title"><img align="center" src="./src/lib/assets/favicon.png" alt="project-image">Neural Chat</h1>

<h2>🚀 Demo</h2>

[http://neuralstudio.intel.com/NeuralChat](http://neuralstudio.intel.com/NeuralChat)

<h2>📸 Project Screenshots:</h2>

<img src="https://imgur.com/04qfWb8.png" alt="project-screenshot" width="800" height="400/">

<img src="https://imgur.com/yZ5AjWE.png" alt="project-screenshot" width="800" height="400/">

<img src="https://imgur.com/OgNJ0cM.png" alt="project-screenshot" width="800" height="400/">


<h2>🧐 Features</h2>

Here're some of the project's features:

- Basic mode：Choose the best model from different domains to chat.
- Advanced mode：Select different models/different knowledge base chats, and you can customize the corresponding parameters.
- hint：Start a chat according to the prompt, which is the first sentence of the chat.
- New Topic: Clear the current context and restart the chat.
- Txt2Img: Generate an image based on the current answer, and you can hover to zoom in the image.
- ChatGPT: Integrated with ChatGPT, you can directly access the relevant functionalities of OpenAI chat by entering your OpenAI API key.
- Chat History: View chat history.
- Regenerate: Regenerates the current conversation.

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

<h2>📕Instructions for using OpenAI:</h2> 

1. Switch to the ChatGPT tab.
<img src="https://imgur.com/8hJW8hh.png" alt="project-screenshot" width="200" height="200/">

2. Enter OpenAI API key in the left sidebar.[Obtain it from https://openai.com/. Without entering the OpenAI API key, you will not be able to use it properly.]
<img src="https://imgur.com/Iu7CmSa.png" alt="project-screenshot" width="200" height="200/">

3. Once you have entered the API key, the input box will be ready for use. All the questions will be answered by ChatGPT.

<h2>Disclaimer</h2> 
This project is to provide instructions and guidance on how to use OpenAI. However, please note the following:

1. OpenAI Policies: OpenAI may have its own policies and regulations, such as API usage limits, pricing plans, service agreements, etc. Please make sure you are aware of and comply with OpenAI's relevant policies to avoid any violations.

2. If you have any questions or issues related to OpenAI while using this service, we take no responsibility for them..