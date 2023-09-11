# Voice Cloning by finetuning a Text-To-Speech(TTS) model

This example shows you how to clone an arbitrary person's voice by finetuning SpeechT5.

## Prepare data

Under this `tts_finetuning` example directory, please make sure there exist two directories: `audios/` and `texts/`, 
where the former contains all the audio files and the latter contains all the texts corresponding to each audio file.

The audio and text file names should be formatted as following:

```
audios/
    1.mp3
    2.mp3
    3.mp3
    ...

texts/
    1.txt
    2.txt
    3.txt
```


Users can use their own audios and corresponding texts, or they can download from the Internet. In this example, we use FeiFei Li's
audio samples extracted from https://github.com/audio-samples/audio-samples.github.io/tree/master/samples/mp3/ted_speakers/FeiFeiLi.

Following the above format, we should rename the audio file names to 1.mp3, 2.mp3 and so on.

Then, we can prepare the texts of those audio files by just listening and writing the texts manually, or running one audio-speech-recognition (ASR) model by using Intel Extension For Transformers ASR interface:

```
python asr.py -i audios/1.mp3 -m openai/whisper-tiny
```

Again, make sure the text file names are formatted as 1.txt, 2.txt and so on.

For simplicity in this example, we have already generated the texts of the audios from https://github.com/audio-samples/audio-samples.github.io/tree/master/samples/mp3/ted_speakers/FeiFeiLi under the `texts/` folder.

## Finetuning

After we have prepare the dataset, we can begin finetuning.

We can just run the following command and the finetuned model is by default named `finetuned_model.pt`.

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
