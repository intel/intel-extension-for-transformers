<h1 align="center" id="title"><img align="center" src="https://imgur.com/ybD9SrK.png" alt="project-image" style="width:60px; height:60px">AI Talking Photo</h1>

<h2>📸 Project Screenshots:</h2>

<img src="https://imgur.com/w6A4cgy.png" alt="project-screenshot" width="400" height="400/">

<img src="https://imgur.com/nriEzQV.png" alt="project-screenshot" width="400" height="400/">

<img src="https://imgur.com/lwVVC5y.png" alt="project-screenshot" width="400" height="400/">


<h2>🧐 Features</h2>

Here're some of the project's features:

- Basic chat: Enter the main page to engage in textual conversations through messaging.
- Upload images: Click the 'Upload' button to select an image from your gallery for uploading. Once the upload is successful, you can simply drag and drop the image into the chat box to start a conversation.
- Photo classify: Successfully uploaded images will be  automatically categorize uploaded images by people, time, and location. Easily access and view different image categories. Customize image information as needed.
- Photo manipulation: You can perform various actions on a single image, such as deleting it, enlarging it, downloading it, resizing it, and rotating it. 
- Hint：Automatically identify image details and show them in a pop-up box. You can then chat with the chatbot using the information and relevant images.
- Txt2Img: Enhance the selected images with artistic transformations, such as creating portraits reminiscent of Van Gogh's style.
- Chat history: The current session will be saved within the specified time frame. If the user closes the window or navigates to other pages, they can resume the conversation from where they left off when they return.
- Talking bot: Supporting voice chat, it allows for the substitution of text with voice in interactions with chatbots, enabling the completion of the aforementioned tasks.
- settings: 
To protect user privacy, we automatically delete stored data in our database every 30 minutes. Users can also manually clear their history or choose not to keep it.

<h2>🛠️ Get it Running:</h2>

1. Clone the repo.

2. cd command to the current folder.

3. Modify the required .env variables.
    ```
    GETIMAGELIST_URL=
    UPLOAD_IMAGE_URL=
    BASE_URL=
    TALKING_URL=
    AUDIO_URL=
    TRAFFIC_URL= 

    ```
4. Execute `npm install` to install the corresponding dependencies.

5. Execute `npm run dev` in both enviroments
