# Workflow purpose
The Huggingface Finetuning(transfer learning) and Inference workflow demonstrates NLP(natural language processing) workflows/pipelines using hugginface transfomer API to be run along with intel optimised software represented using toolkits, domainkits, packages, frameworks and other libraries for effective use of intel hardware leveraging Intel's AI instructions for fast processing and increased performance.The  workflows can be easily used by applications or reference kits showcasing usage. 

The workflow currenly supports
```
Huggingface NLP Finetuning / Transfer Learning
Huggingface NLP Inference
```
The HF Finetuning and Inference workflow supports the following API
```
Huggingface transformer's (trainer API)
Intel's extension for transformers API (Itrex API) also named ( Intel's Transformer/NLP Toolkit)
```

### Architecture
![Reference_Workflow](assets/HFFinetuningAndInference.png)


# Get Started
### Clone this Repository
```
git clone current repository
cd into the current repository directory
```

### Create a new python  (Conda or Venv) environment with env name: "hfftinf_wf"
```shell
conda create -n hfftinf_wf python=3.9
conda activate hfftinf_wf
```
or
```shell
python -m venv hfftinf_wf
source hfftinf_wf/bin/activate
```

### Install package for running hf-finetuning-inference-nlp-workflows
```shell
pip install -r requirements.txt
```

## Running 
See config/README.md for options.
```shell
python src/run.py --config_file config/finetune.yaml 
python src/run.py --config_file config/inference.yaml 
```


