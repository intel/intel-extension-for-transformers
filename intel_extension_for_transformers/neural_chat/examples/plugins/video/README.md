# Face Animation

We optimize SadTalker on Intel Xeon CPU and integrate its face animation functionalities into the video plugin of NeuralChat.

## Prepare Environment

```
conda install ffmpeg
pip install ffmpeg-python
pip install -r requirements.txt
```

## Prepare Models

```
bash download_models.sh checkpoints gfpgan/weights
```

## Usage

### Simply run the test script
```
python main.py
```

### Deploy it as a server

```
neuralchat_server start --config_file face_animation.yaml
```

## Acknowledgements

This plugin is mostly adapted from [SadTalker](https://github.com/OpenTalker/SadTalker). We thank the related authors for their great work!
