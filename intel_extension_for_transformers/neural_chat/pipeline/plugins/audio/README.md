The Audio Processing and Text-to-Speech (TTS) Plugin is a software component designed to enhance audio-related functionality in Neural Chat, specially for TalkingBot. This plugin offers a range of capabilities, primarily focused on processing audio data and converting text into spoken language. Here is a general overview of its key features:

- **Audio Processing**: This component includes a suite of tools and algorithms for manipulating audio data. It can perform tasks such as cut Video, split audio, convert video to audio, noise reduction, equalization, pitch shifting, and audio synthesis, enabling developers to improve audio quality and add various audio effects to their applications.

- **Text-to-Speech (TTS) Conversion**: The TTS plugin can convert written text into natural-sounding speech by synthesizing human-like voices. Users can customize the voice, tone, and speed of the generated speech to suit their specific requirements.

- **Speech Recognition**: The ASR plugin support speech recognition, allowing it to transcribe spoken words into text. This can be used for applications like voice commands, transcription services, and voice-controlled interfaces. It supports both English and Chinese.

- **Multi-Language Support**: The plugin typically supports multiple languages and accents, making it versatile for global applications and catering to diverse user bases. It supports both English and Chinese now.

- **Integration**: Developers can easily integrate this plugin into their applications or systems using APIs.


# Install System Dependency

Ubuntu Command:
```bash
sudo apt-get install ffmpeg
wget http://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2.19_amd64.deb
sudo dpkg -i libssl1.1_1.1.1f-1ubuntu2.19_amd64.deb
```

For other operating systems such as CentOS, you will need to make slight adjustments.

# Multilingual Automatic Speech Recognition (ASR)

We support multilingual Automatic Speech Recognition using Whisper.

## Usage

The AudioSpeechRecognition class provides functionality for converting English/Multilingual audio to text. Here's how to use it:

```python
from intel_extension_for_transformers.neural_chat.pipeline.plugins.audio import AudioSpeechRecognition
# pass the parameter language="auto" to let the asr model automatically detect language
# otherwise, you can pass an arbitrary language to the model (e.g. en/zh/de/fr)
asr = AudioSpeechRecognition("openai/whisper-small", language="auto", device=self.device)
audio_path = "~/audio.wav"  # Replace with the path to your English audio file (supports MP3 and WAV)
result = asr.audio2text(audio_path)
print("ASR Result:", result)
```


# English Text-to-Speech (TTS)

## Usage

The TextToSpeech class in your module provides the capability to convert English text to speech. Here's how to use it:

```python
from intel_extension_for_transformers.neural_chat.pipeline.plugins.audio.tts import TextToSpeech
tts = TextToSpeech()
text_to_speak = "Hello, this is a sample text."  # Replace with your text
output_audio_path = "./output.wav"  # Replace with the desired output audio path
voice = "default"  # You can choose between "default," "pat," or a custom voice
tts.text2speech(text_to_speak, output_audio_path, voice)
```

# Multi Language Text-to-Speech (TTS)

We support multilingual multi-speaker text to speech functionalities (Chinese, English, Japanese) on top of the project [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2), with [IPEX](https://github.com/intel/intel-extension-for-pytorch) BFloat16 inference optimization on Xeon CPU. We finetune our [checkpoints](https://huggingface.co/spycsh/bert-vits-thchs-6-8000) with partial data (6 speakers) from the audio dataset [THCHS-30](https://www.openslr.org/18/).


## Usage

The MultilangTextToSpeech class within your module provides functionality for TTS. Here's how to use it:

```python
from intel_extension_for_transformers.neural_chat.pipeline.plugins.audio.tts_multilang import MultilangTextToSpeech
# Initialize the TTS module
tts = MultilangTextToSpeech()
# Define the text you want to convert to speech
text_to_speak = "欢迎来到英特尔，welcome to Intel。こんにちは！"  # Replace with your Chinese text
# Specify the output audio path
output_audio_path = "./output.wav"  # Replace with your desired output audio path
# Perform text-to-speech conversion
tts.text2speech(text_to_speak, output_audio_path)

# If you want to change the speaker, change the sid
# tts.text2speech(text_to_speak, output_audio_path, sid=1)
```
