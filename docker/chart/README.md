# ITREX Distributed Training

![Version: 0.1.0](https://img.shields.io/badge/Version-0.1.0-informational?style=flat-square) ![Type: application](https://img.shields.io/badge/Type-application-informational?style=flat-square) ![AppVersion: 1.16.0](https://img.shields.io/badge/AppVersion-1.16.0-informational?style=flat-square)

A Helm chart for Kubernetes

## Values

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| distributed.image.image_name | string | `"intel/ai-tools"` |  |
| distributed.image.image_tag | string | `"itrex-devel-1.1.0"` |  |
| distributed.model_name_or_path | string | `"distilbert-base-uncased"` | Name of Model to Train |
| distributed.resources.cpu | int | `4` | Number of CPUs per Pod |
| distributed.resources.memory | string | `"8Gi"` | Amount of Memory per Pod |
| distributed.task_name | string | `"sst2"` | Name of ITREX Task |
| distributed.teacher_model_name_or_path | string | `"textattack/bert-base-uncased-SST-2"` | Name of Huggingface Model to Train off of |
| distributed.workers | int | `4` | Number of Workers (World Size) |
| metadata.name | string | `"itrex-distributed"` |  |
| metadata.namespace | string | `"kubeflow"` |  |
| pvc.name | string | `"itrex"` | Name of PVC for Output Directory |
| pvc.resources | string | `"2Gi"` | Amount of Storage for Output Directory |
| pvc.scn | string | `"nil"` | StorageClassName of PVC |
