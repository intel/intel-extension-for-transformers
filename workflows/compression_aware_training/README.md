# Summary
The workflow provides a generic way to do model compression aware training.

## Compression aware training types supported
1. Distillation (Distill the Finetuned teacher model for your task to a smaller student model) 
2. Quantization Aware Training (QAT) 
3. Distillation followed by Quantization Aware Training (QAT)

### Architecture
![Reference_Workflow](assets/CompressionAwareTraining.png)


## Download Miniconda and install it.
Note: If you have already installed conda on your system, just skip this step.
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
```

## Prepare the conda environment for this workflow
```bash
conda create -n compawaretraining python=3.9.13 --yes
conda activate compawaretraining
```
or
```bash
python -m venv compawaretraining
source compawaretraining/bin/activate
```

### Install package for running compression-aware-training
```bash
sh install.sh
```
or
```bash
pip install -r requirements.txt
```

## Prepare your config file per your requirements
edit config/config.yaml
and set
1. model or the student_model
2. teacher_model
3. dataset
4. task
5. Distillation / Quantization
amongst others specified in the example config.yaml file.

## Run the workflow  using the configurations specified in the yaml file.

### 1. running in vscode
Use the .vscode/launch.json to launch

### 2. running in bash or terminal
```
Student (Current) Model: bert Mini
teacher Model: Bert base pre fined tuned emotion task using HF emotion detection dataset
Task: emotion
output: Distilled Bert Mini, or Distilled and Quantized bert mini
```

Run both traditional distillation followed by Quantization aware training
```bash
python src/run.py config/distillation_with_qat.yaml
```
Run traditional distillation only
```bash
python src/run.py config/distillation.yaml
```
Run Quantization aware training training only
```bash
python src/run.py config/qat.yaml
```

## Run the Workflow using Docker

See [Docker README](./docker/README.md) for more details.

```bash
cd docker
docker compose run dev
```

