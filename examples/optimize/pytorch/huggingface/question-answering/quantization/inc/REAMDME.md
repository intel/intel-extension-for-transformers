[internal]temporary readme to introduce the run command.

```
python run_qa.py     
    --model_name_or_path distilbert-base-uncased-distilled-squad  \
    --dataset_name squad  \
    --tune \   
    --quantization_approach PostTrainingStatic  \
    --do_train \    
    --do_eval \   
    --output_dir ./tmp/squad_output \
    --overwrite_output_dir
```

