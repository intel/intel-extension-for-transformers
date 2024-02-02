
This README is intended to guide you through setting up the service of audio plugins using the NeuralChat framework. You can deploy it on various platforms, including Intel XEON Scalable Processors, Habana's Gaudi processors (HPU), Intel Data Center GPU and Client GPU, Nvidia Data Center GPU and Client GPU.

# Introduction
NeuralChat provides services not only based on LLM, but also support single service of plugin such as audio. This example introduce our solution to build a plugin-as-service server. Though few lines of code, our api could help the user build a audio plugin service, which is able to transfer between texts and audios.

Before deploying this example, please follow the instructions in the [README](../../README.md) to install the necessary dependencies.

# Setup Environment

## Setup Conda

First, you need to install and configure the Conda environment:

```shell
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda*.sh
source ~/.bashrc
```

## Install numactl

Next, install the numactl library:

```shell
sudo apt install numactl
```

## Install Python dependencies

Install the following Python dependencies using Conda:

```shell
conda install astunparse ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses -y
conda install jemalloc gperftools -c conda-forge -y
```

Install other dependencies using pip:

```bash
pip install -r ../../../requirements.txt
```


# Configure YAML

You can customize the configuration file 'audio_service.yaml' to match your environment setup. Here's a table to help you understand the configurable options:

|  Item                             | Value                                  |
| --------------------------------- | ---------------------------------------|
| host                              | 127.0.0.1                              |
| port                              | 7777                                   |
| device                            | "auto"                                 |
| asr.enable                        | true                                   |
| asr.args.device                   | "cpu"                                  |
| asr.args.model_name_or_path       | "openai/whisper-small"                 |
| asr.args.bf16                     | false                                  |
| tts.enable                        | true                                   |
| tts.args.device                   | "cpu"                                  |
| tts.args.voice                    | "default"                              |
| tts.args.stream_mode              | false                                  |
| tts.args.output_audio_path        | "./output_audio.wav"                   |
| tts.args.speedup                  | 1.0                                    |
| tasks_list                        | ['plugin_audio']              |


# Run the audio service server
To start the audio service server, run the following command:

```shell
nohup bash run.sh &
```

# Call the audio plugin service
To call the started audio service, the APIs are listed as follows:
1. http://127.0.0.1:7777/plugin/audio/asr , upload an audio file and return the text contents.
2. http://127.0.0.1:7777/plugin/audio/tts , input text string and return the binary content of the audio.
3. http://127.0.0.1:7777/plugin/audio/create_embedding, upload an audio file and create an embedding of your voice.
