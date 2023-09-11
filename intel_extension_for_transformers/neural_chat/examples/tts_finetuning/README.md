# Voice Cloning by finetuning a Text-To-Speech (TTS) model

This example shows you how to clone an arbitrary person's voice by finetuning SpeechT5.

## Prepare data

Under this `tts_finetuning` example directory, please make sure there exist two directories: `audios/` and `texts/`, 
where the former contains all the audio files and the latter contains all the texts corresponding to each audio file.

The audio and text file names should be formatted like following (We just need to make sure every audio file has the same name with its text file):

```
audios/
    <audio_name_0>.mp3
    <audio_name_1>.mp3
    <audio_name_2>.mp3
    ...

texts/
    <audio_name_0>.txt
    <audio_name_1>.txt
    <audio_name_2>.txt
    ...
```


Users can use their own audios and corresponding texts, or they can download from the Internet. Here are the [audio samples](https://github.com/audio-samples/audio-samples.github.io/tree/master/samples/mp3/ted_speakers/FeiFeiLi) that we use in this example.

Then, we can prepare the texts of those audio files by just listening and writing the texts manually, or running one audio-speech-recognition (ASR) model by using Intel Extension For Transformers ASR interface:

```python
# Replace <xxxxx_sample-0> with your input audio name
python asr.py -i audios/<xxxxx_sample-0>.mp3 -m openai/whisper-tiny
```

For simplicity in this example, we have already generated the texts of the aforementioned audio samples under the `texts/` folder.

## Finetuning

After preparing the dataset, we can start finetuning. We can just run the following command and the finetuned model is by default named `finetuned_model.pt`.

```
python finetune.py
```

You can change the arguments in `TTSDatasetArguments` or `TTSModelArguments` in the `finetune.py` passed to Intel Extension For Transformers TTS interface to customize your finetuning process.

## Inference

Now we have our finetuned model `finetuned_model.pt`, so let us check the quality and performance of that. We can run the following command:
```
python inference.py
```

Then you will see a prompt in the console `Write a sentence to let the talker speak:`, and you can enter your input text in the console and press ENTER on your keyboard to generate the speech with your finetuned voice.
