from transformers import TrainingArguments
from transformers import logging as hf_logging
import argparse

import os, sys
print('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]) + "/src")
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]) + "/src")

from infer_itrex import ItrexInfer

hf_logging.set_verbosity_info()

data = {
        'args': {
            'model_name_or_path': 'output_dir',
            'tokenizer_name': 'output_dir',
            'dataset': 'local',
            'local_dataset': {
                'inference_input': [
                    '/data/datac/samanway/annotation/annotation.csv',
                    '/data/datac/samanway/annotation/d1.csv',
                    '/data/datac/samanway/annotation/d2.csv'
                ],
                'delimiter': ',',
                'features': {
                    'class_label': 'label',
                    'data_column': 'symptoms',
                    'id': 'Patient_ID'
                },
                'label_list': ['Malignant', 'Normal', 'Benign']
            },
            'infer_impl': 'itrex',
            'dtype_inf': 'fp32',
            'max_seq_len': 64,
            'smoke_test': False,
            'max_train_samples': None,
            'max_test_samples': None,
            'preprocessing_num_workers': 8,
            'overwrite_cache': True,
            'inference_output': 'inference_predictions_report.yaml',
            'multi_instance': False
        },
        'training_args': {
            'do_predict': True,
            'per_device_eval_batch_size': 100,
            'output_dir': './output_dir'
        }
    }

training_args = TrainingArguments(output_dir="./output_dir")
for item in data["training_args"]:
    setattr(training_args, item, data["training_args"][item])

parser = argparse.ArgumentParser()
args = parser.parse_args()

for item in data["args"]:
    setattr(args, item, data["args"][item])

kwargs = {"args": args, "training_args": training_args}        

infer = ItrexInfer(**kwargs)        
infer.e2e_infer_setup_only()

inf_list = [
    '/data/datac/samanway/annotation/annotation.csv',
    '/data/datac/samanway/annotation/d1.csv',
    '/data/datac/samanway/annotation/d2.csv'
]
            
for f in inf_list:
    infer.e2e_infer_only(f)