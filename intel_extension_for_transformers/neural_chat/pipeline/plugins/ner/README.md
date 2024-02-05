<div align="center">
<h1>NER: Named Entity Recognition</h3>
<div align="left">

# 🏠Introduction
The Named Entity Recognition(NER) Plugin is a software component designed to enhance NER-related functionality in Neural Chat. NER is a subtask of information extraction that seeks to locate and classify named entities mentioned in unstructured text into pre-defined categories such as person names, organizations, locations, medical codes, time expressions, quantities, monetary values, percentages, etc.

This plugin provides large language models(llm) with different precision, the supported precisions are listed below:

1. Float32 (FP32)
2. BFloat16 (BF16)
3. INT8
4. INT4

As the precision decreases, the llm model inference time decreases, but so does the accuracy of the inference results. You can choose models with different accuracies to accomplish NER inferencing tasks according to your needs.

# 🔧Install dependencies
Before using NER plugin, install dependencies below. You need to download spacy model before using it, choose proper one according to Spacy Official Documents [here](https://spacy.io/usage/models) to meet your demands. The example model here is `en_core_web_lg`.

```bash
# install required packages
pip install -r requirements.txt
# download spacy model
python -m spacy download en_core_web_lg
```


# 🚀Usage
With different model configurations, the NER plugins of varying precisions have two ways of usage as below.

## Inference with FP32/BF16

```python
from intel_extension_for_transformers.neural_chat.pipeline.plugins.ner import NamedEntityRecognition
ner_obj = NamedEntityRecognition
# modify the query here for customized NER task
query = "Show me photos taken in Shanghai today."
result = ner_obj.inference(query=query) # add argument "bf16=True" for BF16 inference
print("NER result: ", result)
```

## Inference with INT8/INT4

```python
from ner_int import NamedEntityRecognitionINT
# set compute_dtype='int8' and weight_dtype='int4' for INT4 inference
ner_obj = NamedEntityRecognitionINT(compute_dtype='fp32', weight_dtype='int8')
query = "Show me photos taken in Shanghai today."
result = ner_obj.inference(query=query)
print("NER result: ", result)
```

# 🚗Parameters
## Plugin Parameters
You can costomize the NER inference parameters to meet the personal demands for better performance. You can set the specific parameter by `plugins.ner.args["xxx"]`. Below are the descriptions of each available parameters.
```python
model_name_or_path [str]: The huggingface model name or local path of the downloaded llm model. Default to "./neural-chat-7b-v3-1/".

spacy_model [str]: The Spacy model for NLP process, specify it according to the downloaded Spacy model. Default to "en_core_web_lg".

bf16 [bool]: Choose whether to use BF16 precision for NER inference. Default to False.
```
As for INT8 and INT4 model the plugin parameters are slightly different. You can set the specific parameter by `plugins.ner_int.args["xxx"]`.
```python
model_name_or_path [str]: The huggingface model name or local path of the downloaded llm model. Default to "./neural-chat-7b-v3-1/".

spacy_model [str]: The Spacy model for NLP process, specify it according to the downloaded Spacy model. Default to "en_core_web_lg".

compute_dtype [str]: The dtype of model while computing. Set to "int8" for INT4 inference for better performance. Default to "fp32".

weight_dtype [str]: The dtype of model weight. Set to "int4" for INT4 inference. Default to "int8".
```

## Inference Parameters
### FP32/BF16 Inference
```python
query [str]: The query string that needs NER to extract entities. Mandatory parameter that must be passed.

prompt [str]: The inference prompt for llm model. You could construct customized prompt for certain demands. Default to 'construct_default_prompt' in '/neural_chat/pipeline/plugins/ner/utils/utils.py'.

max_new_tokens [int]: The max generated token numbers. Default to 32.

temperature [float]: The temperature of llm. Default to 0.01.

top_k [int]: The top_k parameter of llm. Default to 3.

repetition_penalty [float]: The repetition penalty of llm. Default to 1.1.
```

### INT8/INT4 Inference
```python
query [str]: The query string that needs NER to extract entities. Mandatory parameter that must be passed.

prompt [str]: The inference prompt for llm model. You could construct customized prompt for certain demands. Default to 'construct_default_prompt' in '/neural_chat/pipeline/plugins/ner/utils/utils.py'.

threads [int]: The thread number of model inference. Set to the core number of your server for minimal inferencing time. Default to 52.

max_new_tokens [int]: The max generated token numbers. Default to 32.

seed [int]: The random seed of llm. Default to 1234.
```
