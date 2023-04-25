# Image classification
The script `run_image_classification.py` provides three quantization approaches (PostTrainingStatic, PostTrainingStatic and QuantizationAwareTraining) based on [Intel® Neural Compressor](https://github.com/intel/neural-compressor).

Here is how to run the script:
1. quantization with PostTrainingStatic

```
sh run_tuning.sh

# if you want to try other approaches, please revise run_tuning.sh.
```

2. evaluation with int8 model (please remind the model path in the script)
```
run run_benchmark.sh --int8
```

3. evaluation with fp32 model
```
run run_benchmark.sh
```


### Validated model list

|Dataset|Pretrained model|PostTrainingDynamic | PostTrainingStatic | QuantizationAwareTraining
|---|------------------------------------|---|---|---
|imagenet-1k|google/vit-base-patch16-224| ✅| ✅| N/A|
