Step-by-Step​
============
The script `run_fewshot_setfit.py` provides the fewshot training of SetFit.

# Prerequisite​
## Create Environment​
Recommend python 3.9 or higher version.
```shell
pip install intel-extension-for-transformers
pip install -r requirements.txt
```
>**Note**: Suggest use transformers no higher than 4.34.1


# Run
To train and evaluate SetFit on 8 examples (per class) on the sst2 dataset, use below command:

```bash
python run_fewshot_setfit.py
```

Above command use default logistic regression classifier, if you desire to use torch's linear classifier, use below command:

```bash
python run_fewshot_setfit.py --classifier pytorch
```