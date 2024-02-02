# Face Animation

We optimize SadTalker on Intel Xeon CPU and integrate its face animation functionalities into the video plugin of NeuralChat.

## Prepare Environment

```
cd intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/pipeline/plugins/video/face_animation
conda install ffmpeg
pip install ffmpeg-python
pip install -r requirements.txt
```

## Usage

```python
from sadtalker import SadTalker
device = "cuda" if torch.cuda.is_available() else "cpu"
sadtalker = SadTalker(device=device)
# without enhancer
sadtalker.convert(source_image=source_image, driven_audio=driven_audio, output_video_path="./response.mp4",
                bf16=True, result_dir="./results", p_num=4, enhancer=None)
# with enhancer
sadtalker.convert(source_image=source_image, driven_audio=driven_audio, output_video_path="./response.mp4",
                bf16=True, result_dir="./results", p_num=4, enhancer='gfpgan')
```

## Acknowledgements

This plugin is mostly adapted from [SadTalker](https://github.com/OpenTalker/SadTalker). We thank the related authors for their great work!
