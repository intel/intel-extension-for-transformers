NeuralChat Fine-tuning
============

This example demonstrates how to finetune the pretrained generative image-to-text model on customized dataset.

# Prerequisite​

## 1. Environment​
### Bare Metal
Recommend python 3.9 or higher version.
```shell
pip install -r requirements.txt
pip install transformers==4.34.1
# To use ccl as the distributed backend in distributed training on CPU requires to install below requirement.
python -m pip install oneccl_bind_pt==2.2.0 -f https://developer.intel.com/ipex-whl-stable-cpu
```
>**Note**: Suggest using transformers no higher than 4.34.1

### Docker 
Pick either one of below options to setup docker environment.
#### Option 1 : Build Docker image from scratch
Please refer to this section : [How to build docker images for NeuralChat FineTuning](../../../docker/finetuning/README.md#21-build-docker-image) to build docker image from scratch.  

#### Option 2: Pull existing Docker image
Please follow the session [itrex docker setup](../../../docker/finetuning/README.md#22-docker-pull-from-docker-hub) and use the docker pull command to pull itrex docker image.  


Once you have the docker image ready, please follow [run docker image](../../../docker/finetuning/README.md#3-create-docker-container) session to launch a docker instance from the image.   


## 2. Prepare the Model

#### microsoft/git-base
To acquire the checkpoints and tokenizer, the user can get those files from [microsoft/git-base](https://huggingface.co/microsoft/git-base).
Users could follow below commands to get the checkpoints from github repository after the access request to the files is approved.
```bash
git lfs install
git clone https://huggingface.co/microsoft/git-base
```

## 3. Prepare Dataset

For datasets exist in the Hugging Face Hub, user can use `dataset_name` argument to pass in the needed dataset.
For local datasets, user can follow this [guide](https://huggingface.co/docs/datasets/v2.18.0/en/image_dataset#image-captioning) from datasets' official document to create a metadata file that contain image and text pairs, than use `train_dir` and optionally `validation_dir` to pass in the path to the needed dataset.

### Dataset related arguments
- **dataset_name**: The name of the dataset to use (via the datasets library).
- **dataset_config_name**: The configuration name of the dataset to use (via the datasets library).
- **train_dir**: A folder containing the training data.
- **validation_dir**: A folder containing the validation data.
- **image_column**: The column of the dataset containing an image or a list of images.
- **caption_column**: The column of the dataset containing a caption or a list of captions.
- **validation_split_percentage**: The percentage of the train set used as validation set in case there's no validation split.

# Finetune

Use the below command line for finetuning `microsoft/git-base` model on the `gaodrew/roco-65k-256px` dataset.

```bash
python finetune_clm.py \
        --model_name_or_path "microsoft/git-base" \
        --bf16 True \
        --dataset_name "gaodrew/roco-65k-256px" \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --do_train \
        --learning_rate 1e-4 \
        --num_train_epochs 3 \
        --logging_steps 100 \
        --save_total_limit 2 \
        --overwrite_output_dir \
        --log_level info \
        --save_strategy epoch \
        --output_dir ./git-base_finetuned_model \
        --task image2text \
        --full_finetune \
        --bits 16
