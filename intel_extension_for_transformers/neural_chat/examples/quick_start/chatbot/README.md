
# Build Your Chatbot with Intel® Extension for Transformers neural-chat

# 1 Setup Environment

## 1.1 Install intel-extension-for-transformers

```
conda create -n itrex-chatbot python=3.9
conda activate itrex-chatbot
pip install intel-extension-for-transformers==1.3.2
```
## 1.2 Install neural-chat dependency

```
pip install accelerate
pip install transformers_stream_generator

git clone https://github.com/intel/intel-extension-for-transformers.git ~/itrex
cd ~/itrex
git checkout v1.3.2

cd ~/itrex/intel_extension_for_transformers/neural_chat
```

Setup CPU platform go to [1.2.1](#121-cpu-platform)

Setup GPU platform go to [1.2.2](#122-GPU-Platform)

### 1.2.1 CPU Platform
`pip install -r requirements_cpu.txt`

Got to [Section 2](#2-Run-the-chatbot-in-command-mode).

### 1.2.2 GPU Platform

#### prerequisite
GPU driver and oneAPI 2024.0 is required.

`pip install -r requirements_xpu.txt`

# 2 Run the chatbot in command mode

## Usage

Go back to the quick_example folder and run the example

```
source /opt/intel/oneapi/setvars.sh
python chatbot.py
```

```
/home/xiguiwang/anaconda3/envs/test/lib/python3.9/site-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension: ''If you don't plan on using image functionality from `torchvision.io`, you can ignore this warning. Otherwise, there might be something wrong with your environment. Did you have `libjpeg` or `libpng` installed before building `torchvision` from source?
  warn(
2024-03-20 11:22:33,191 - datasets - INFO - PyTorch version 2.1.0a0+cxx11.abi available.
2024-03-20 11:22:33,191 - datasets - INFO - TensorFlow version 2.16.1 available.
/home/xiguiwang/anaconda3/envs/test/lib/python3.9/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
Loading model Intel/neural-chat-7b-v3-1
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.77it/s]
2024-03-20 11:22:38,398 - root - INFO - Model loaded.
Once upon a time, a little girl lived in a quaint village nestled among rolling hills. She had a heart filled with curiosity and dreams of adventure. One day, she decided to leave her cozy home behind and set out on a journey to explore the world beyond her familiar surroundings.

As she ventured forth, she encountered many wondrous sights and met fascinating people along the way. The little girl learned about different cultures, customs, and traditions that broadened her perspective and enriched her life. Her experiences taught her valuable lessons about kindness, courage, and resilience.

Throughout her travels, she made lifelong friends who shared her passion for discovery. Together, they faced challenges and celebrated triumphs, forming unbreakable bonds that would last a lifetime.

Eventually, the little girl returned to her village, now a wise and compassionate young woman. She brought back knowledge and memories that inspired others to dream big and follow their hearts. As she grew older, she continued to share her stories and wisdom with those around her, inspiring future generations to embrace the beauty of the unknown and never stop seeking new horizons.

```

# 3. Run chatbot in server mode with UI

## 3.1 Start the service

```
python chatbot_server.py
```

Here is the completely output:
```
(/home/xiguiwang/ws2/conda/itrex-rag) xiguiwang@icx02-tce-atsm:~/ws2/AI-examples/chatbot$ python chatbot_server.py /home/xiguiwang/ws2/conda/itrex-rag/lib/python3.9/site-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension: ''If you don't plan on using image functionality from `torchvision.io`, you can ignore this warning. Otherwise, there might be something wrong with your environment. Did you have `libjpeg` or `libpng` installed before building `torchvision` from source?
  warn(
2024-03-18 16:28:01,454 - datasets - INFO - PyTorch version 2.1.0a0+cxx11.abi available.
/home/xiguiwang/ws2/conda/itrex-rag/lib/python3.9/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
Loading model Intel/neural-chat-7b-v3-1
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.55it/s]
2024-03-18 16:28:08,634 - root - INFO - Model loaded.
Loading config settings from the environment...
INFO:     Started server process [1544268]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)

```


### 3.1.1 Verify the client connection to server is OK.

Open a new linux console, run following command

`curl -vv -X POST http://127.0.0.1:8000/v1/chat/completions`

Check the output. Make sure there is no network connection and proxy setting issue at Client side

### 3.1.2 Test request command at client side

Sent a request to chatbat-server from client 


```
curl http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "Intel/neural-chat-7b-v3-1",
    "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me about Intel Xeon Scalable Processors."}
    ]
    }'
```

At the server side, there is message:
```
INFO:     127.0.0.1:52532 - "POST /v1/chat/completions HTTP/1.1" 200 OK
```

At the client side, the response are similar message as following.
The message contains the LLM answer and other information about the request.
```
{"id":"chatcmpl-29GVLhfoSJHeHTgqL4HgxP","object":"chat.completion","created":1710750809,"model":"Intel/neural-chat-7b-v3-1","choices":[{"index":0,"message":{"role":"assistant","content":"Intel Xeon Scalable Processors are a series of high-performance central processing units (CPUs) designed for data centers, cloud computing, and other demanding computing environments. They are part of Intel's Xeon family of processors, which are specifically tailored for server and workstation applications.\n\nThe Xeon Scalable Processors were introduced in 2017 and are based on Intel's Skylake microarchitecture. They offer significant improvements in performance, efficiency, and scalability compared to their predecessors. These processors are available in various configurations, including single-socket, dual-socket, and multi-socket systems, catering to different workloads and requirements.\n\nSome key features of Intel Xeon Scalable Processors include:\n\n1. Scalable performance: The processors can be configured to meet specific workload needs, allowing for better resource utilization and improved performance.\n\n2. High-speed memory support: They support up to 6 channels of DDR4 memory, enabling faster data transfer and improved system performance.\n\n3. Advanced security features: The processors come with built-in security features, such as Intel Software Guard Extensions (SGX), which help protect sensitive data and applications from potential threats.\n\n4. Enhanced virtualization capabilities: The Xeon Scalable Processors are designed to support multiple virtual machines, making them suitable for virtualized environments.\n\n5. Improved energy efficiency: The processors are designed to optimize power consumption, reducing operational costs and minimizing environmental impact.\n\nOverall, Intel Xeon Scalable Processors are a powerful and versatile choice for organizations seeking high-performance computing solutions in data centers, cloud environments, and other demanding applications."},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"total_tokens":0,"completion_tokens":0}}
```

## 3.2 Set up Server mode UI

Create UI conda envitonment
```
conda create -n chatbot-ui python=3.9
conda activate chatbot-ui

cd ~/itrex/intel_extension_for_transformers/neural_chat/ui/gradio/basic
pip install -r requirements.txt

pip install gradio==3.36.0
pip install pydantic==1.10.13
```

## 3.3 Start the web service

Set the default service port
Edit app.py line 745, set the server port. For example we set port as 8008.

```
    demo.queue(
        concurrency_count=concurrency_count, status_update_rate=10, api_open=False
    ).launch(
        server_name=host, server_port=8008, share=share, max_threads=200
    )
```

Start the service:
`python app.py`

The output is as following:
```
/home/xiguiwang/ws2/conda/chatbot-ui/lib/python3.9/site-packages/gradio_client/documentation.py:103: UserWarning: Could not get documentation group for <class 'gradio.mix.Parallel'>: No known documentation group for module 'gradio.mix'
  warnings.warn(f"Could not get documentation group for {cls}: {exc}")
/home/xiguiwang/ws2/conda/chatbot-ui/lib/python3.9/site-packages/gradio_client/documentation.py:103: UserWarning: Could not get documentation group for <class 'gradio.mix.Series'>: No known documentation group for module 'gradio.mix'
  warnings.warn(f"Could not get documentation group for {cls}: {exc}")
2024-03-27 11:00:24 | INFO | gradio_web_server | Models: ['Intel/neural-chat-7b-v3-1']
2024-03-27 11:00:26 | ERROR | stderr | sys:1: GradioDeprecationWarning: The `style` method is deprecated. Please set these arguments in the constructor instead.
2024-03-27 11:00:26 | INFO | stdout | Running on local URL:  http://0.0.0.0:8008
2024-03-27 11:00:26 | INFO | stdout |
2024-03-27 11:00:26 | INFO | stdout | To create a public link, set `share=True` in `launch()`.
```

The log shows the service is started on prot 8008.
You can access chatbot through web browser on port 8008 now.
