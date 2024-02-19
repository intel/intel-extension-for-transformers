
# Multi-Modal

Large Language and Vision Assistant (LLaVA) is a multi-modal training framework that proposed from [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) and [Improved Baselines with Visual Instruction Tuning](https://arxiv.org/abs/2310.03744). This example demonstrates how to train mult-modal model on Intel Gaudi2.

## Validated Model List
|Pretrained model| LLaVA | 
|------------------------------------|---|
|Mistral series| ✅|
|LLaMA series| ✅|

**Note:** For Salesforce/codegen25-7b-* series models same with LLaMA architecture, need install `pip install transformers==4.33.2` refer [this](https://github.com/salesforce/CodeGen/issues/82)

## Train

LLaVA training consists of two stages: (1) feature alignment stage: use our 558K subset of the LAION-CC-SBU dataset to connect a *frozen pretrained* vision encoder to a *frozen LLM*; (2) visual instruction tuning stage: use 150K GPT-generated multimodal instruction-following data, plus around 515K VQA data from academic-oriented tasks, to teach the model to follow multimodal instructions.

### Pretraining

##### Prepare data
Download the 558K subset of the LAION-CC-SBU dataset with BLIP captions from [liuhaotian/LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) in `./pretraining_data`

##### Training 

Training script with DeepSpeed ZeRO-2: `scripts/pretrain.sh`.

- `--mm_projector_type mlp2x_gelu`: the two-layer MLP vision-language connector.
- `--vision_tower openai/clip-vit-large-patch14-336`: CLIP ViT-L/14 336px.
- `--use_habana, --use_lazy_mode` for Intel Gaudi2 setting.

**Note:** If don't set `--use_habana, --use_lazy_mode`, the code can also run on gpus as well.

### Visual Instruction Tuning

##### Prepare data

Please download the annotation of the final mixture our instruction tuning data [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_v1_5_mix665k.json), and download the images from constituting datasets:

- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
- GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing), **we save all files as `.jpg`**
- TextVQA: [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

After downloading all of them, organize the data as follows in `./finetuning_data`,

```
├── coco
│   └── train2017
├── gqa
│   └── images
├── ocr_vqa
│   └── images
├── textvqa
│   └── train_images
└── vg
    ├── VG_100K
    └── VG_100K_2
```

##### Start training!

Training script with DeepSpeed ZeRO-3: `scripts/finetune.sh`, and lora has been enabled by running `scripts/finetune_lora.sh`


New options to note:

- `--mm_projector_type mlp2x_gelu`: the two-layer MLP vision-language connector.
- `--vision_tower openai/clip-vit-large-patch14-336`: CLIP ViT-L/14 336px.
- `--image_aspect_ratio pad`: this pads the non-square images to square, instead of cropping them; it slightly reduces hallucination.
- `--group_by_modality_length True`: this should only be used when your instruction tuning dataset contains both language (e.g. ShareGPT) and multimodal (e.g. LLaVA-Instruct). It makes the training sampler only sample a single modality (either image or language) during training, which we observe to speed up training by ~25%, and does not affect the final outcome.
- `--use_habana, --use_lazy_mode` for Intel Gaudi2 setting.
- For finetuning stage, when using Intel Gaudi2, `--pad_max True` should be set, which will pad input sequence length (text + image patches) to `--model_max_length`.

**Note:** If don't set `--use_habana, --use_lazy_mode`, the code can also run on gpus as well.


## MMMU Evaluation on Gaudi2

```
# For static shape, not support beam search currently
bash scripts/mmmu_eval.sh
```
*note: if you want to do evaluation on GPUs, please refer [the original code](https://github.com/MMMU-Benchmark/MMMU/tree/main)*
