# Language modeling
The scripts `run_clm.py`, `run_mlm.py` and `run_plm.py` provides three quantization approaches (PostTrainingStatic, PostTrainingStatic and QuantizationAwareTraining) based on [Intel® Neural Compressor](https://github.com/intel/neural-compressor).

Here is how to run the scripts:

**Causal Language modeling (CLM)**

```
python run_clm.py \     
    --model_name_or_path EleutherAI/gpt-neo-125M \     
    --dataset_name wikitext \     
    --dataset_config_name wikitext-2-raw-v1 \     
    --tune \    
    --quantization_approach PostTrainingStatic \          
    --do_train \     
    --do_eval \     
    --output_dir ./tmp/clm_output \ 
    --overwrite_output_dir

```

**Masked Language modeling (MLM)**

```
python run_mlm.py \     
    --model_name_or_path bert-base-uncased \  
    --dataset_name wikitext \     
    --dataset_config_name wikitext-2-raw-v1 \     
    --tune \    
    --quantization_approach PostTrainingStatic \          
    --do_train \     
    --do_eval \     
    --output_dir ./tmp/mlm_output \ 
    --overwrite_output_dir
```

**Permutation Language modeling (PLM)**

```
    python run_plm.py     
    --model_name_or_path xlnet-base-cased  
    --dataset_name wikitext     
    --dataset_config_name wikitext-2-raw-v1     
    --tune    
    --quantization_approach PostTrainingStatic          
    --do_train     
    --do_eval     
    --output_dir ./tmp/plm_output
    --overwrite_output_dir

```

### Validated model list

|Type|Pretrained model|PostTrainingDynamic | PostTrainingStatic | QuantizationAwareTraining
|---|------------------------------------|---|---|---
|CLM|EleutherAI/gpt-neo-125M| ✅| ✅| N/A
|MLM|bert-base-uncased| ✅| ✅| N/A
|PLM|xlnet-base-cased| ✅| ✅| N/A

### Command

```
bash run_tuning.sh  --topology=topology
```

```
bash run_benchmark.sh --topology=topology --mode=benchmark