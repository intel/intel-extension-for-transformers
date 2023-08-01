python run_tango.py \
    --small_model_name_or_path philschmid/tiny-bert-sst2-distilled \
    --big_model_name_or_path textattack/roberta-base-SST-2 \
    --task_name sst2 \
    --confidence_threshold 0.9 