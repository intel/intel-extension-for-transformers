# Finetune Peppa Pig Voice Example

![peppa](https://upload.wikimedia.org/wikipedia/en/thumb/8/86/Peppa_Pig_logo.svg/1280px-Peppa_Pig_logo.svg.png)

In this repo, we would like to introduce how to finetune Peppa Pig's Voice by using Intel® Extension for Transformers NeuralChat Audio plugin.

The audios are small clips extracted from https://www.youtube.com/watch?v=RkM0hTmPGc8

To finetune pig peppa's model, please run

```python
python finetune_peppa.py
```

To generate the speech using your finetuned model, please run

```python
python inference_peppa.py
```

Then you should see in the console

```
Write a sentence to let peppa pig speak:
```

Then you can enter your text in the console and press ENTER on your keyboard to generate the speech.

Here is one example output audio with the input text "Hello, I am peppa pig. Nice to meet you.":

<audio controls="" preload="none"><source src="output_sample.wav"></audio>