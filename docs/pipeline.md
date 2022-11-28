# Pipeline

1. [Introduction](#introduction)
2. [Examples](#examples)

    2.1. [Pipeline Inference for INT8 Model](#pipeline-inference-for-int8-model)

    2.2. [Pipeline Inference for Executor Backend](#pipeline-inference-for-executor-backend)

## Introduction
The pipeline is inherited from [huggingface/transformers pipeline](https://github.com/huggingface/transformers/blob/main/docs/source/en/pipeline_tutorial.mdx), it is simple to use any model from [Hub](https://huggingface.co/models) for inference on any language, computer vision, speech, and multimodal tasks. Two features for int8 model inference and model inference on [executor backend](../intel_extension_for_transformers/backends/neural_engine/) have been added to the extension.


## Examples

### Pipeline Inference for INT8 Model

1. Initialize a pipeline instance with a model name and specific task.
    ```py
    from intel_extension_for_transformers.optimization.pipeline import pipeline
    text_classifier = pipeline(
        task="text-classification",
        model="Intel/distilbert-base-uncased-finetuned-sst-2-english-int8-static",
        framework="pt",
        device=torch.device("cpu"),
    )
    ```
2. Pass your input text to the pipeline instance for inference.
    ```py
    outputs = text_classifier("This is great !")
    # output: [{'label': 1, 'score': 0.9998425245285034}]
    ```


### Pipeline Inference for Executor Backend

For executor, we only accept ONNX model now for pipeline. Users can get ONNX model from PyTorch model with our existing [API](export.md). Right now, pipeline for executor only supports text-classification task. 

1. Initialize a pipeline instance with an ONNX model, model config, model tokenizer and specific backend. The MODEL_NAME is the pytorch model name you used for exporting the ONNX model.
    ```py
    from intel_extension_for_transformers.optimization.pipeline import pipeline
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    text_classifier = pipeline(
        task="text-classification",
        config=config,
        tokenizer=tokenizer,
        model='fp32.onnx',
        model_kwargs={'backend': "executor"},
    )
    ```

2. Pass your input text to the pipeline instance for inference.
    ```py
    outputs = text_classifier(
        "But believe it or not , it 's one of the most "
        "beautiful , evocative works I 've seen ."
    )
    # output: [{'label': 'POSITIVE', 'score': 0.9998886585235596}]
    ```

