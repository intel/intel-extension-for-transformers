# Compression Aware

![Version: 0.1.0](https://img.shields.io/badge/Version-0.1.0-informational?style=flat-square) ![Type: application](https://img.shields.io/badge/Type-application-informational?style=flat-square) ![AppVersion: 1.16.0](https://img.shields.io/badge/AppVersion-1.16.0-informational?style=flat-square)

A Helm chart for Kubernetes

## Values

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| dataset.nfs.path | string | `"nil"` | Path to Local NFS Share in Cluster Host |
| dataset.nfs.server | string | `"nil"` | Hostname of NFS Server |
| dataset.nfs.subPath | string | `"nil"` | Path to dataset in Local NFS |
| dataset.s3.key | string | `"nil"` | Path to Dataset in S3 Bucket |
| dataset.type | string | `"<nfs/s3>"` | `nfs` or `s3` dataset input enabler |
| image.name | string | `"intel/ai-workflows"` |  |
| metadata.name | string | `"compression-aware"` |  |
| proxy | string | `"nil"` |  |
| workflow.config | string | `"qat"` | name of config file. Presets are `qat`, `distillation`, `distillation_with_qat` and `config` |
