Hugging Face Space helps to make some amazing ML applications more accessible to the community. Inspired by this, we can create a chatbot application on Hugging Face Spaces. Alternatively, you can also deploy the frontend on your own server.

# Deploy on Huggingface Space

## 🚀 Create a new space on Huggingface
To create a new application space on Hugging Face, visit the website at [https://huggingface.co/new-space](https://huggingface.co/new-space) and follow the below steps to create a new space.

![Hugging Face Spaces](https://i.imgur.com/ZODwuWt.png)

![Create New Space](https://i.imgur.com/QyjqUd6.png)

The new space is like a new project that supports GitHub-style code repository management.

## 🚀 Check configuration
We recommend using Gradio as the Space SDK, keeping the default values for the other settings.

For detailed information about the configuration settings, please refer to the [Hugging Face Spaces Config Reference](https://huggingface.co/docs/hub/spaces-config-reference).

## 🚀 Setup application
We strongly recommend utilizing the provided textbot frontend code as it represents the reference implementation already deployed on Hugging Face Space. To establish your application, simply copy the code files from this directory and adjust their configurations as necessary. Alternatively, you have the option to clone the existing space from [NeuralChat-ICX-INT4](https://huggingface.co/spaces/Intel/NeuralChat-ICX-INT4).

![Clone Space](https://i.imgur.com/76N8m5B.png)

You can also choose to update the URLs of two backend services in the `app.py` file, corresponding to two chatbots.

![Update backend URL](https://i.imgur.com/F0FLeEn.png)

# Deploy on your server

Before deploying on your server, you need to install and configure the Conda environment:

## Setup Conda

First, you need to install and configure the Conda environment:

```shell
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda*.sh
source ~/.bashrc
```

## Install Python dependencies

Install the required Python dependencies using pip:

```shell
pip install -r requirements.txt
```

## Run the frontend

Launch the chatbot frontend on your server using the following command:

```shell
nohup python app.py &
```

This will run the chatbot application in the background on your server.

Once the application is running, you can find the access URL in the trace log:

```log
INFO | gradio_web_server | Models: Intel/neural-chat-7b-v3-1
INFO | stdout | Running on local URL:  http://0.0.0.0:7860
```
since there are two services, start two backends and generate URLs for both backends.

The URLs to access the chatbot frontend are http://{SERVER_IP_ADDRESS_1}:80 and http://{SERVER_IP_ADDRESS_2}:80, respectively. Please remember to replace {SERVER_IP_ADDRESS} with your server's actual IP address.


![URL](https://i.imgur.com/FDKSnIo.png)

You also have the option to update the backend service URL in the `app.py` file.

![Update backend URL](https://i.imgur.com/j7TTYaW.png)

>**Note**: Please use Gradio version 3.34.0.
