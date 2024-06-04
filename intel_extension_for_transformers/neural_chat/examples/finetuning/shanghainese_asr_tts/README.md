
# Shanghainese ASR (Audio-Speech-Recognition) and TTS (Text-To-Speech) finetuning/inference

This example introduces how to do Shanghainese audio-to-text and text-to-audio conversion.


## Related models

### ASR

* Conversion from the Shanghainese audio to the Shanghainese text
Finetuned [jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn)

* Conversion from Shanghainese text to Mandarin text
Finetuned [Helsinki-NLP/opus-mt-zh-en](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en)


### TTS

* Conversion from Mandarin text to Shanghainese text  
Finetuned [Helsinki-NLP/opus-mt-en-zh](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh)

* Conversion from Shanghainese text to Shanghainese audio  
Finetuned [VITS](https://github.com/jaywalnut310/vits)

## Prepare Environment

## 1. Install requirements

```sh
pip install -r requirements.txt
# force to install torch 2.0+
pip install torch==2.1.0
```

<!-- ## 2. Build monotonic alignment search

```py
cd monotonic_align
mkdir monotonic_align
python setup.py build_ext --inplace
cd ..
``` -->

## Finetuning

### ASR

#### 1. Prepare the data

Please check this [repo](https://github.com/Cosmos-Break/asr) and download the folders named like `Shanghai_*` to the current directory.

#### 2. Do finetuning of the Shanghainese Audio -> Shanghainese text ASR model

```py
python train_asr.py
```

#### 3. Do finetuning of the Shanghainese text -> Mandarian text translation model

```py
python train_translation.py
```

### TTS

#### 1. Download the pre-finetuned VITS model

Download [model](https://sjtueducn-my.sharepoint.com/:u:/g/personal/cjang_cjengh_sjtu_edu_cn/EfnEO6kW-CNNhywJmIZNPU0BUmFdSArguFETp0pjtvHZBA?e=dKJULk) (2796 epochs)

Put the `model.pth` model under `model/`.


#### 3. Finetune the Mandarian text -> Shanghainese text translation model

```py
python train_translation_reverse.py
```

## Inference

### ASR

#### Do inference of the Shanghainese Audio -> Shanghainese text ASR model


```
python inference_asr.py
```

#### Do inference of the Shanghainese text -> Mandarian text translation model

```
python inference_translation.py
```

### TTS


#### Do inference of the Mandarian text -> Shanghainese text translation model
```
python inference_translation_reverse.py
```

#### Do inference of the Shanghainese text -> Shanghainese audio TTS model

```
python inference_tts.py
```

## Demo

```sh
export no_proxy="localhost,127.0.0.1"
nohup python -u app.py &
```

![asr-sh](https://imgur.com/dAB4vxj.png)

![tts-sh](https://imgur.com/0i0xcVH.png)

## Acknowledgements

The code is adapted from [Cosmos-Break/asr](https://github.com/Cosmos-Break/asr) and [CjangCjengh/vits](https://github.com/CjangCjengh/vits). We thanks the authors for their great work!
