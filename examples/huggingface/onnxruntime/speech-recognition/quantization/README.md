Step-by-Step​
============
The script `run_whisper.py` provides two quantization approaches (PostTrainingStatic and PostTrainingDynamic) based on [Intel® Neural Compressor](https://github.com/intel/neural-compressor) with [LibriSpeech test-clean](https://huggingface.co/datasets/librispeech_asr) dataset.

# Prerequisite​
## 1. Create Environment​
```shell
pip install -r requirements.txt
```

## 2. Prepare Model<200b>
```
optimum-cli export onnx --model openai/whisper-large whisper-large-with-past/ --task automatic-speech-recognition-with-past --opset 13
```

# Run
## 1. Quantization

- To get int8 model

```
bash run_tuning.sh --config=openai/whisper-large \
                   --dataset_location=/path/to/dataset \ # optional
                   --input_model=whisper-large-with-past/ \
                   --output_model=whisper-large-with-past-static/ \ # or whisper-large-with-past-dynamic
                   --approach=static # or dynamic
```

- To get model accuracy

```
bash run_benchmark.sh --config=whisper-large-with-past \
                      --dataset_location=/path/to/dataset \ # optional
                      --input_model=whisper-large-with-past-static/ \
                      --int8 \
                      --mode=accuracy
```

- To get model performance

```
numactl -m 0 -C 0-3 bash run_benchmark.sh --config=whisper-large-with-past \
                                          --dataset_location=/path/to/dataset \ # optional
                                          --input_model=whisper-large-with-past-static/ \
                                          --mode=benchmark \
                                          --iters=100 \
                                          --cores_per_instance=4 \
                                          --int8 \
                                          --max_new_tokens=16
```

**Notes**: 
 - If users don't set dataset_location, it will download the dataset or use the cached dataset automatically.
 - numactl command is used to bind specific cores.

# Validated model list

|Topology|Pretrained model|PostTrainingDynamic|PostTrainingStatic
|---|------------------------------------|---|---
|whisper_large|openai/whisper-large| ✅| ✅|


