# TangoBERT

This code reproduces results from [TangoBERT paper](http://arxiv.org/abs/2204.06271).

TangoBERT is a cascaded model architecture in
which instances are first processed by an efficient but less accurate first tier model, and
only part of those instances are additionally
processed by a less efficient but more accurate
second tier model. The decision of whether to
apply the second tier model is based on a confidence score produced by the first tier model. 
Instance are first
processed by the efficient first tier model. If this model
is confident regarding its prediction, it produces the final prediction. Otherwise, the more accurate second
tier model processes the instance and produces the final prediction.

![TangoBERT architecture](system.png "TangoBERT architecture")
## Installation

```bash
pip install -r requirements.txt
```

## Running on GLUE tasks

```bash
pyhon run_tango.py --task_name TASK-NAME --small_model_name_or_path SMALL-MODEL --big_model_name_or_path BIG-MODEL 
                  --layer_num TRANSFORMER-LAYERS 
                  [--device_small DEVICE-SMALL] [--device_big DEVICE-BIG]
                  [--per_device_eval_batch_size-big BATCH-SIZE-SMALL] [--per_device_eval_batch_size-big BATCH-SIZE-BIG] 
                  [--confidence_threshold THRESHOLD]
                  
```

* `TASK-NAME`: name of the GLUE task.
* `SMALL-MODEL`: path to the small pretrained model or model identifier from huggingface.co/models.
* `BIG-MODEL`: path to the big pretrained model or model identifier from huggingface.co/models.
* `DEVICE-SMALL`: defines the device (e.g., "cpu", "cuda:1", "mps", or a GPU ordinal rank like 1) on which small model pipeline will be allocated.
* `DEVICE-BIG`: defines the device (e.g., "cpu", "cuda:1", "mps", or a GPU ordinal rank like 1) on which big model pipeline will be allocated.
* `BATCH-SIZE-SMALL`: batch size (per device) for small model inference.
* `BATCH-SIZE-BIG`: batch size (per device) for big model inference.
* `THRESHOLD`: confidence threshold of small model prediction.



### SST-2
```bash
python run_tango.py 
    --small_model_name_or_path philschmid/tiny-bert-sst2-distilled 
    --big_model_name_or_path textattack/roberta-base-SST-2 
    --task_name sst2
    --confidence_threshold 0.9
```

## Citation

```
@inproceedings{mamou2022tangobert,
    title={TangoBERT: Reducing Inference Cost by using Cascaded Architecture},
    author={Jonathan Mamou and Oren Pereg and Moshe Wasserblat and Roy Schwartz},
    booktitle = "Energy Efficient Training and Inference of Transformer Based Models workshop, AAAI Conference on Artificial Intelligence,
    url = "http://arxiv.org/abs/2204.06271",
    year = {2023}
}
```