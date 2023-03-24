Step-by-Step
=========
This document describes the end-to-end workflow for Text-to-image generative AI models [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) and [runwayml/stable-diffusion-v1-5](https://github.com/runwayml/stable-diffusion) across the Neural Engine backend.

# Prerequisite

## 1. Installation
### 1.1 Install python environment
Create a new python environment
```shell
conda create -n <env name> python=3.8
conda activate <env name>
```
Make sure you have the autoconf installed. 
Also, `gcc` higher than 9.0, `cmake` higher than 3 is required.
```shell
gcc -v
cmake --version
conda install cmake
sudo apt install autoconf
```
Install Intel® Extension for Transformers, please refer to [installation](https://github.com/intel/intel-extension-for-transformers/blob/main/docs/installation.md)
```shell
# Install from pypi
pip install intel-extension-for-transformers

# Install from source code
cd <intel_extension_for_transformers_folder>
git submodule update --init --recursive
python setup.py install
```
Install required dependencies for examples
```shell
cd <intel_extension_for_transformers_folder>/examples/deployment/neural_engine/stable_diffusion
pip install -r requirements.txt
```
>**Note**: Recommend install protobuf <= 3.20.0 if use onnxruntime <= 1.11


### 1.2 Environment variables preload libjemalloc.so can improve the performance when multi instances.
```
export LD_PRELOAD=<intel_extension_for_transformers_folder>/intel_extension_for_transformers/backends/neural_engine/executor/third_party/jemalloc/lib/libjemalloc.so
```
Using weight sharing can save memory and improve the performance when multi instances.
```
export WEIGHT_SHARING=1
export INST_NUM=<inst num>
```
## 2. End-to-End Workflow of the Pretrained Model

The stable diffusion mainly includes three onnx models: text_encoder, unet, vae_decoder.

The pretrained model [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) and [runwayml/stable-diffusion-v1-5](https://github.com/runwayml/stable-diffusion) provied by diffusers are be the same in the default config.

Here we take CompVis/stable-diffusion-v1-4 as an example.

### 2.1 Get models
Export FP32 ONNX models from the hugginface diffusers module, command as follows:

```python
python prepare_model.py --input_model=CompVis/stable-diffusion-v1-4 --output_path=./model
```

By setting --bf16 to export FP32 and BF16 models.
```python
python prepare_model.py --input_model=CompVis/stable-diffusion-v1-4 --output_path=./model --bf16
```

### 2.2 Compile Models
Export three FP32 onnx sub models of the stable diffusion to Nerual Engine IRs.

```bash
# just running the follow bash comand to get all IRs.
bash export_model.sh --input_model=model --precision=fp32
```

If you want to export models seperately, command as follows:
```python
# 1. text encoder
python export_ir.py --onnx_model=./model/text_encoder_fp32/model.onnx --pattern_config=text_encoder_pattern.conf --output_path=./fp32_ir/text_encoder/

# 2. unet
python export_ir.py --onnx_model=./model/unet_fp32/model.onnx --pattern_config=unet_pattern.conf --output_path=./fp32_ir/unet/

# 3. vae_decoder
python export_ir.py --onnx_model=./model/vae_decoder_fp32/model.onnx --pattern_config=vae_decoder_pattern.conf --output_path=./fp32_ir/vae_decoder/
```

Export three BF16 onnx sub models of the stable diffusion to Nerual Engine IRs.

```bash
# just running the follow bash comand to get all IRs.
bash export_model.sh --input_model=model --precision=bf16
```

If you want to export models seperately, command as follows:
```python
# 1. text encoder
python export_ir.py --onnx_model=./model/text_encoder_bf16/model.onnx --pattern_config=text_encoder_pattern.conf --output_path=./bf16_ir/text_encoder/

# 2. unet
python export_ir.py --onnx_model=./model/unet_bf16/model.onnx --pattern_config=unet_pattern.conf --output_path=./bf16_ir/unet/

# 3. vae_decoder
python export_ir.py --onnx_model=./model/vae_decoder_bf16/bf16-model.onnx --pattern_config=vae_decoder_pattern.conf --output_path=./bf16_ir/vae_decoder/
```

## 2.3 Run Models

Text-to-image: using one sentence to create a picture!

```python
# Running FP32 models or BF16 models, just import differnt IRs.
# FP32 models
GLOG_minloglevel=2 python run_executor.py --ir_path=./fp32_ir

# BF16 models
GLOG_minloglevel=2 python run_executor.py --ir_path=./bf16_ir
```
> Note: 
> 1. The default pretrained model is "CompVis/stable-diffusion-v1-4".
> 2. The default prompt is "a photo of an astronaut riding a horse on mars" and the default output name is "astronaut_rides_horse.png".
> 3. The ir directory should include three IRs for text_encoder, unet and vae_decoder.

## Benchmark

### 2.1 Performance

Python API command as follows:
```shell
GLOG_minloglevel=2 python run_executor.py --ir_path=./fp32_ir --mode=performance
GLOG_minloglevel=2 python run_executor.py --ir_path=./bf16_ir --mode=performance
```

### 2.2 Accuracy
Frechet Inception Distance(FID) metric is used to evaluate the accuracy. This case we check the FID scores between the pytorch image and engine image.

By setting --accuracy to check FID socre.
Python API command as follows:
```shell
GLOG_minloglevel=2 python run_executor.py --ir_path=./fp32_ir --mode=accuracy
GLOG_minloglevel=2 python run_executor.py --ir_path=./bf16_ir --mode=accuracy
```
