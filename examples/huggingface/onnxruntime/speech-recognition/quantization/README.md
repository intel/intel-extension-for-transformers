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

- To get int4 model

```
bash run_tuning.sh --config=openai/whisper-large \
                   --dataset_location=/path/to/dataset \ # optional
                   --input_model=whisper-large-with-past/ \
                   --output_model=whisper-large-onnx-int4/ \
                   --approach=weight_only
```

## 2. Benchmark
- To get model accuracy

```
bash run_benchmark.sh --config=whisper-large-with-past \
                      --dataset_location=/path/to/dataset \ # optional
                      --input_model=whisper-large-with-past-static/ \
                      --int8 \ # or int4
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
                                          --int8 \ # or int4
                                          --max_new_tokens=16
```

**Notes**: 
 - If users don't set dataset_location, it will download the dataset or use the cached dataset automatically.
 - numactl command is used to bind specific cores.

## 3. Audio inference
- To run audio sample inference with FP32/INT8 (both static and dynamic) models

```
bash run_audio_inference.sh --config=openai/whisper-large \ # model_name_or_path
                            --audio_path=/path/to/dataset \ # optional, support .wav, .mp3 and other ffmpeg supported formats
                            --input_model=whisper-large-with-past-static/ \ # folder path of onnx model
```

- To run audio sample inference with INT4 models

Upgrade onnxruntime to 1.16.0 first and then:

```
bash run_audio_inference.sh --config=openai/whisper-tiny \ # model_name_or_path
                            --audio_path=/path/to/dataset \ # optional, support .wav, .mp3 and other ffmpeg supported formats
                            --input_model=whisper-tiny-onnx-int4/ \ # folder path of onnx model
```

Available INT4 models on huggingface:
[whisper-tiny](https://huggingface.co/Intel/whisper-tiny-onnx-int4), 
[whisper-base](https://huggingface.co/Intel/whisper-base-onnx-int4), 
[whisper-small](https://huggingface.co/Intel/whisper-small-onnx-int4), 
[whisper-medium](https://huggingface.co/Intel/whisper-medium-onnx-int4), 
[whisper-large](https://huggingface.co/Intel/whisper-large-onnx-int4), 
[whisper-large-v2](https://huggingface.co/Intel/whisper-large-v2-onnx-int4).

**Notes**: 
 - If users don't set audio_path, it will use sample.wav in intel_extension_for_transformers/neural_chat/assets/audio folder for test.

# Validated model list

|Topology|Pretrained model|PostTrainingDynamic|PostTrainingStatic|WeightOnly4Bit|
|---|------------------------------------|---|---|---
|whisper_tiny|openai/whisper-tiny| | | ✅|
|whisper_base|openai/whisper-base| | | ✅|
|whisper_small|openai/whisper-small| | | ✅|
|whisper_medium|openai/whisper-medium| | | ✅|
|whisper_large|openai/whisper-large| | | ✅|
|whisper_large_v2|openai/whisper-large-v2| ✅| ✅| ✅|
