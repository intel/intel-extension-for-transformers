This README is intended to guide you through setting up the backend for a voice chatbot using the NeuralChat framework. You can deploy this text chatbot on various platforms, including Intel XEON Scalable Processors, Habana's Gaudi processors (HPU), Intel Data Center GPU and Client GPU, Nvidia Data Center GPU and Client GPU.


# Setup Conda

First, you need to install and configure the Conda environment:

```shell
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda*.sh
source ~/.bashrc
```

# Install numactl

Next, install the numactl library:

```shell
sudo apt install numactl
```

# Install Python dependencies

Install the following Python dependencies using Conda:

```shell
conda install astunparse ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses -y
conda install jemalloc gperftools -c conda-forge -y
conda install -q -y pyg -c pyg
conda install -q -y pytorch cudatoolkit=11.3 -c pytorch
```

Install other dependencies using pip:

```bash
pip install -r ../../../requirements.txt
pip install -U torch torchaudio --no-cache-dir
```

# Configure the voicebot.yaml

You can customize the configuration file 'textbot.yaml' to match your environment setup. Here's a table to help you understand the configurable options:

|  Item                | Value                                      |
| --------------------- | --------------------------------------- |
| host                  | 127.0.0.1                                |
| port                  | 8888                                     |
| model_name_or_path    | "Intel/neural-chat-7b-v3-1"          |
| device                | "cpu"                                    |
| asr.enable            | true                                     |
| asr.args.device       | "cpu"                                    |
| asr.args.model_name_or_path | "openai/whisper-small"             |
| asr.args.bf16         | false                                    |
| tts.enable            | true                                     |
| tts.args.device       | "cpu"                                    |
| tts.args.voice        | "default"                                |
| tts.args.stream_mode  | true                                     |
| tts.args.output_audio_path | "./output_audio"                    |
| tts.args.speedup      | 1.0                                      |
| tasks_list            | ['voicechat']                            |



# Run the VoiceChat server
To start the VoiceChat server, use the following command:

```shell
nohup bash run.sh &
```

# Quick test with OpenAI compatible endpoints (audio)

To make our audio service compatible to OpenAI [endpoints](https://platform.openai.com/docs/api-reference/audio/), we offer the following three endpoints:

```
/v1/audio/speech
/v1/audio/transcriptions
/v1/audio/translations
```

To test whether the talkingbot server can serve your requests correctly, you can use `curl` as follows:

```
curl http://localhost:8888/v1/audio/translations \
  -H "Content-Type: multipart/form-data" \
  -F file="@sample_1.wav"

curl http://localhost:8888/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F file="@sample_zh_cn.wav"

curl http://localhost:8888/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "speecht5",
    "input": "The quick brown fox jumped over the lazy dog.",
    "voice": "default"
  }' \
  --output speech.mp3
```

# Customized endpoints of a audio-input-audio-output pipeline

You can check `intel_extension_for_transformers/neural_chat/server/restful/voicechat_api.py` to see the other customized endpoints offered by us:

```
/v1/talkingbot/asr
/v1/talkingbot/llm_tts
/v1/talkingbot/create_embedding
```

`/v1/talkingbot/asr` is equivalent to `/v1/audio/transcriptions` and for backward compatibility we simply keep that for audio-to-speech conversion.
`/v1/talkingbot/llm_tts` merges two tasks: `LLM text generation` and the `text to speech` into one process, which is designed specifically for converting steadily the LLM streaming outputs to speech.
`/v1/talkingbot/create_embedding` is used to create a SpeechT5 speaker embedding for zero-shot voice cloning. Although voice-cloning is relatively weak for SpeechT5, we still keep this endpoint for quick start. If you want to clone your voice properly, please check the current best practices for SpeechT5 based on few-shot voice-cloning finetuning in this [repo](../../../../finetuning/tts/).
