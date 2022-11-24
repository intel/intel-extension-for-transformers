# Pipelines for inference

The pipeline is inherited from transformers [pipeline](https://github.com/huggingface/transformers/blob/main/docs/source/en/pipeline_tutorial.mdx), and two more features are appended.

* Use a [`pipeline`] for int8 model inference.
* Use a [`pipeline`] for inference on our [executor](../intel_extension_for_transformers/backends/neural_engine/) backend.

Executor is a inference tool for accelerated deployment in Intel_Extension_for_Transformers.

## Pipeline usage

----
### **INT8 model**

1. Initialize a pipeline instance with model name and specific task.
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

----

### **Executor backend**

For executor, we only accept ONNX model now for pipeline. Users can get onnx model from PyTorch model with our existing [API](export.md). Right now, pipeline for executor only supports text-classcification task. 

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

