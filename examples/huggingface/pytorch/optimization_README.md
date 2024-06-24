# Huggingface Examples

Welcome to Pytorch Huggingface examples. The examples is following from [Huggingface transformers with PyTorch backend](https://github.com/huggingface/transformers/tree/main/examples/pytorch) and model compressor technology is dependent on [Intel® Neural Compressor](https://github.com/intel/neural-compressor), including quantization, pruning and distillation. 

## Quantization approach

| Task | dynamic | static | qat
|---|:---:|:---:|:---:|
|**`language-modeling`**| ✅ | ✅ | ✅
|**`multi-choice`**| ✅ | ✅ | ✅
|**`question-answering`**| ✅ | ✅ | ✅
|**`text-classification`**| ✅ | ✅ | ✅
|**`token-classification`**| ✅ | ✅ | ✅
|**`summarization`**| ✅ | ✅ | ✅
|**`translation`**| ✅ | ✅ | ✅

## Pruning approach

| Task | Magnitude
|---|:---:|
|**`question-answering`**| ✅ 
|**`text-classification`**| ✅

## Distillation

| Task | Knowledge Distillation 
|---|:---:|
|**`language-modeling`**| N/A  
|**`question-answering`**| ✅
|**`text-classification`**| ✅

## Orchestrate

| Task | Knowledge Distillation | Magnitude Pruning | Pattern Lock Pruning
|---|:---:|:---:| :---:|
|**`question-answering`**| ✅ | ✅ | ✅
|**`text-classification`**| ✅| ✅ | ✅
