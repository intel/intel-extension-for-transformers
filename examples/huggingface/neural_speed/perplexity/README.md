# Perplexity
This document describes the perplexity evaluation of customizable configurations using ITREX over neural-speed.

## Prepare Python Environment
```bash
conda create -n <env name> python=3.10 gxx
conda activate <env name>
```

Install Intel® Extension for Transformers, please refer to [installation](/docs/installation.md).
```bash
# Install from pypi
pip install intel-extension-for-transformers==1.3.1

# Or, install from source code
cd <intel_extension_for_transformers_folder>
pip install -r requirements.txt
pip install -v .
```

Install required dependencies for this example including neural-speed (Note that you may need some specific version of
transformers when running some models)
```bash
pip install -r requirements.txt
```
## Prepare Dataset
Taking wikitext-2 and pg19 as examples:
``` python
import datasets
dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split='test', num_proc=16)
dataset.save_to_disk('~/wikitext-2-raw-v1-data-test')
dataset = datasets.load_dataset("pg19", split='test', num_proc=16)
dataset.save_to_disk('~/pg19-data-test')
```

## Evaluate Perplexity
```bash
python perplexity.py --model_name model_path_or_hugging_face_model_name --dataset_name path_to_local_dataset
```
For more details, please check docopt generated with `python perplexity.py --help`.
