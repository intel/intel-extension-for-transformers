# Huggingface Examples

Welcome to Pytorch Huggingface examples. The examples is following from [Huggingface transformers with PyTorch backend](https://github.com/huggingface/transformers/tree/main/examples/pytorch) and model compressor technology is dependend on [Intel® Neural Compressor](https://github.com/intel/neural-compressor), including quantization, pruning and distillation. 

## Quantization approach

| Task | PostTrainingDynamic | PostTrainingStatic | QuantizationAwareTraining
|---|:---:|:---:|:---:|
|**`language-modeling`**| ✅ | ✅ | N/A
|**`multi-choice`**| ✅ | ✅ | N/A
|**`question-answering`**| ✅ | ✅ | N/A
|**`text-classification`**| ✅ | ✅ | ✅
|**`token-classification`**| ✅ | ✅ | N/A
|**`summarization`**| ✅ | N/A | N/A
|**`translation`**| ✅ | N/A | N/A

## Pruning approach

| Task | Magnitude
|---|:---:|
|**`question-answering`**| ✅ 
|**`text-classification`**| ✅

## Distillation

| Task | Knowledge Distillation | AutoDistillation
|---|:---:|:---:|
|**`language-modeling`**| N/A | ✅ 
|**`question-answering`**| ✅ | N/A
|**`text-classification`**| ✅| N/A

## Orchestrate

| Task | Knowledge Distillation | Magnitude Pruning | Pattern Lock Pruning
|---|:---:|:---:|:---:| 
|**`question-answering`**| ✅ | ✅ | ✅
|**`text-classification`**| ✅| ✅ | ✅
