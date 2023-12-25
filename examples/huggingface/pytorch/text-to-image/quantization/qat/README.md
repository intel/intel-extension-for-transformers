Step-by-step
============

In this example we demonstrate how to apply quantization aware training as well as knowledge distillation on the UNet of Stable Diffusion. Script ```train_text_to_image_qat.py``` is based on [huggingface/diffusers](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image) and provides quantization aware training with knowledge distillation approach based on [Intel® Neural Compressor](https://github.com/intel/neural-compressor).

# Prerequisite​

## Create Environment
```
pip install intel-extension-for-transformers
pip install -r requirements.txt
pip install transformers==4.34.1
```
>**Note**: Please use transformers no higher than 4.34.1


## Prepare Dataset
We use 20000 images from the [LAION-400M dataset](https://laion.ai/blog/laion-400-open-dataset/) to do quantization aware training with knowledge distillation for Stable Diffusion. Please follow the steps in [here](https://laion.ai/blog/laion-400-open-dataset/) to download the 20000 image and text pairs and put these files in a folder. Then we need to create metadata for this small dataset use the below command.
```bash
python create_metadata.py --image_path /path/to/image_text_pairs_folder
```


# Run

## Training
Use below command to run quantization aware training with knowledge distillation for UNet.
```bash
accelerate launch train_text_to_image_qat.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --train_data_dir /path/to/image_text_pairs_folder \
    --use_ema --resolution=512 --center_crop --random_flip \
    --train_batch_size=4 --gradient_accumulation_steps=1 \
    --gradient_checkpointing --max_train_steps=300 \
    --learning_rate=1e-05 --max_grad_norm=1 --lr_scheduler="constant" \
    --lr_warmup_steps=0 --output_dir="sdv1-5-qat_kd" --use_cpu
```

## Inference

Once the training is done, you can use below command to generate images with quantized UNet.
```bash
python text2images.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --quantized_model_name_or_path sdv1-5-qat_kd/quant_model.pt \
    --captions "a photo of an astronaut riding a horse on mars"
```

You can also use BF16 UNet for inference on some steps of denoising loop instead of INT8 UNet to improve output images quality, to do so, just add `--use_bf16` argument in the above command.

Below are two results comparison of fp32 model, int8 model and mixture of bf16 model and int8 model. Note int8 model is trained on an Intel® Xeon® Platinum 8480+ Processor.
<br>
With caption `"a photo of an astronaut riding a horse on mars"`, results of fp32 model, int8 model and mixture of bf16 model and int8 model are listed left, middle and right respectively.
<br>
<img src="./fp32 images/a photo of an astronaut riding a horse on mars fp32.png" width = "300" height = "300" alt="FP32" align=center />
<img src="./int8 images/a photo of an astronaut riding a horse on mars int8.png" width = "300" height = "300" alt="INT8" align=center />
<img src="./int8 bf16 images/a photo of an astronaut riding a horse on mars int8 bf16.png" width = "300" height = "300" alt="INT8 BF16" align=center />

With caption `"The Milky Way lies in the sky, with the golden snow mountain lies below, high definition"`, results of fp32 model, int8 model and mixture of bf16 model and int8 model are listed left, middle and right respectively.
<br>
<img src="./fp32 images/The Milky Way lies in the sky, with the golden snow mountain lies below, high definition fp32.png" width = "300" height = "300" alt="FP32" align=center />
<img src="./int8 images/The Milky Way lies in the sky, with the golden snow mountain lies below, high definition int8.png" width = "300" height = "300" alt="INT8" align=center />
<img src="./int8 bf16 images/The Milky Way lies in the sky, with the golden snow mountain lies below, high definition int8 bf16.png" width = "300" height = "300" alt="INT8 BF16" align=center />

## FID evaluation
We have also evaluated FID scores on COCO2017 validation dataset for FP32 model, BF16 model, INT8 model and mixture of BF16 and INT8 model. FID results are listed below.

| Precision            | FP32  | BF16  | INT8  | INT8+BF16 |
|----------------------|-------|-------|-------|-----------|
| FID on COCO2017 val  | 30.48 | 30.58 | 35.46 | 30.63     |

To evaluated FID score on COCO2017 validation dataset for mixture of BF16 and INT8 model, you can use below command.

```bash
python evaluate_fid.py \
    --model_name_or_path runwayml/stable-diffusion-v1-5 \
    --int8_model_path sdv1-5-qat_kd/quant_model.pt \
    --dataset_path /path/to/COCO2017 \
    --output_dir ./output_images \
    --precision int8-bf16
```
