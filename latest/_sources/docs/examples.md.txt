# Examples

1. [Quantization](#Quantization)

    1.1 [Stock PyTorch Examples](#Stock-PyTorch-Examples)

    1.2 [Intel Extension for Pytorch (IPEX) Examples](#Intel-Extension-for-Pytorch-IPEX-Examples)

    1.3 [Intel TensorFlow Examples](#Intel-TensorFlow-Examples)

2. [Length Adaptive Transformers](#Length-Adaptive-Transformers)

3. [Pruning](#Pruning)

4. [Distillation](#Distillation)

    4.1 [Knowledge Distillation](#Knowledge-Distillation)

    4.2 [Auto Distillation (NAS Based)](#Auto-Distillation-NAS-Based)

5. [Orchestrate](#Orchestrate)

6. [Reference Deployment on Neural Engine](#Reference-Deployment-Neural-Engine)

   6.1 [Dense Reference](#Dense-Reference-Deployment-Neural-Engine)

   6.2 [Sparse Reference](#Sparse-Reference-Deployment-Neural-Engine)



Intel Extension for Transformers is a powerful toolkit with multiple model optimization techniques for Natural Language Processing Models, including quantization, pruning, distillation, auto distillation and orchestrate. Meanwhile Intel Extension for Transformers provides Transformers-accelerated Neural Engine, an optimized backend for NLP models to demonstrate the deployment.

## Quantization
### Stock PyTorch Examples
<table>
<thead>
  <tr>
    <th>Model</th>
    <th>Task</th>
    <th>Dataset</th>
    <th>PostTrainingDynamic</th>
    <th>PostTrainingStatic</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td><a href="https://huggingface.co/EleutherAI/gpt-j-6B">gpt-j-6B</a></td>
    <td>language-modeling(CLM)</td>
    <td><a href="https://huggingface.co/datasets/wikitext">wikitext</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/sysresearch101/t5-large-finetuned-xsum-cnn">t5-large-finetuned-xsum-cnn</a></td>
    <td>summarization</td>
    <td><a href="https://huggingface.co/datasets/cnn_dailymail">cnn_dailymail</a></td>
    <td>&#10004;</td>
    <td> </td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/flax-community/t5-base-cnn-dm">t5-base-cnn-dm</a></td>
    <td>summarization</td>
    <td><a href="https://huggingface.co/datasets/cnn_dailymail">cnn_dailymail</a></td>
    <td>&#10004;</td>
    <td> </td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/lambdalabs/sd-pokemon-diffusers">lambdalabs/sd-pokemon-diffusers</a></td>
    <td>text-to-image</td>
    <td>image</td>
    <td>&#10004;</td>
    <td>&#10004;</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/bert-base-uncased">bert-base-uncased</a></td>
    <td>language-modeling(MLM)</td>
    <td><a href="https://huggingface.co/datasets/wikitext">wikitext</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/xlnet-base-case">xlnet-base-cased</a></td>
    <td>language-modeling(PLM)</td>
    <td><a href="https://huggingface.co/datasets/wikitext">wikitext</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/EleutherAI/gpt-neo-125M">EleutherAI/gpt-neo-125M</a></td>
    <td>language-modeling(CLM)</td>
    <td><a href="https://huggingface.co/datasets/wikitext">wikitext</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/sshleifer/tiny-ctrl">sshleifer/tiny-ctrl</a></td>
    <td>language-modeling(CLM)</td>
    <td><a href="https://huggingface.co/datasets/wikitext">wikitext</a></td>
    <td>WIP :star:</td>
    <td>&#10004;</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/ehdwns1516/bert-base-uncased_SWAG">ehdwns1516/bert-base-uncased_SWAG</a></td>
    <td>multiple-choice</td>
    <td><a href="https://huggingface.co/datasets/swag">swag</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/distilbert-base-uncased-distilled-squad">distilbert-base-uncased-distilled-squad</a></td>
    <td>question-answering</td>
    <td><a href="https://huggingface.co/datasets/squad">SQuAD</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/valhalla/longformer-base-4096-finetuned-squadv1">valhalla/longformer-base-4096-finetuned-squadv1</a></td>
    <td>question-answering</td>
    <td><a href="https://huggingface.co/datasets/squad">SQuAD</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/lvwerra/pegasus-samsum">lvwerra/pegasus-samsum</a></td>
    <td>summarization</td>
    <td><a href="https://huggingface.co/datasets/samsum">samsum</a></td>
    <td>&#10004;</td>
    <td>WIP :star:</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/textattack/bert-base-uncased-MRPC">textattack/bert-base-uncased-MRPC</a></td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/glue/viewer/mrpc/train">MRPC</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/echarlaix/bert-base-uncased-sst2-acc91.1-d37-hybrid">echarlaix/bert-base-uncased-sst2-acc91.1-d37-hybrid</a></td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/glue/viewer/sst2/train">SST-2</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english">distilbert-base-uncased-finetuned-sst-2-english</a></td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/glue/viewer/sst2/train">SST-2</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/elastic/distilbert-base-uncased-finetuned-conll03-english">elastic/distilbert-base-uncased-finetuned-conll03-english</a></td>
    <td>token-classification</td>
    <td><a href="https://huggingface.co/datasets/conll2003">conll2003</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/t5-small">t5-small</a></td>
    <td>translation</td>
    <td><a href="https://huggingface.co/datasets/wmt16">wmt16</a></td>
    <td>&#10004;</td>
    <td>WIP :star:</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/Helsinki-NLP/opus-mt-en-ro">Helsinki-NLP/opus-mt-en-ro</a></td>
    <td>translation</td>
    <td><a href="https://huggingface.co/datasets/wmt16">wmt16</a></td>
    <td>&#10004;</td>
    <td>WIP :star:</td>
  </tr>
</tbody>
</table>

<table>
<thead>
  <tr>
    <th>Model</th>
    <th>Task</th>
    <th>Dataset</th>
    <th>QuantizationAwareTraining</th>
    <th>No Trainer quantization</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td><a href="https://huggingface.co/textattack/bert-base-uncased-MRPC">textattack/bert-base-uncased-MRPC</a></td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/glue/viewer/mrpc/train">MRPC</a></td>
    <td>&#10004;</td>
    <td> </td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/echarlaix/bert-base-uncased-sst2-acc91.1-d37-hybrid">echarlaix/bert-base-uncased-sst2-acc91.1-d37-hybrid</a></td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/glue/viewer/sst2/train">SST-2</a></td>
    <td> </td>
    <td>&#10004;</td>
  </tr>
</tbody>
</table>

### Intel Extension for Pytorch (IPEX) examples
<table>
<thead>
  <tr>
    <th>Model</th>
    <th>Task</th>
    <th>Dataset</th>
    <th>PostTrainingStatic</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td><a href="https://huggingface.co/distilbert-base-uncased-distilled-squad">distilbert-base-uncased-distilled-squad</a></td>
    <td>question-answering</td>
    <td><a href="https://huggingface.co/datasets/squad">SQuAD</a></td>
    <td>&#10004;</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/bert-large-uncased-whole-word-maskinuned-squad">bert-large-uncased-whole-word-maskinuned-squad</a></td>
    <td>question-answering</td>
    <td><a href="https://huggingface.co/datasets/squad">SQuAD</a></td>
    <td>&#10004;</td>
  </tr>
</tbody>
</table>

### Intel TensorFlow Examples
<table>
<thead>
  <tr>
    <th>Model</th>
    <th>Task</th>
    <th>Dataset</th>
    <th>PostTrainingStatic</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td><a href="https://huggingface.co/bert-base-cased-finetuned-mrpc">bert-base-cased-finetuned-mrpc</a></td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/glue/viewer/mrpc/train">MRPC</a></td>
    <td>&#10004;</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/xlnet-base-cased">xlnet-base-cased</a></td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/glue/viewer/mrpc/train">MRPC</a></td>
    <td>&#10004;</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/distilgpt2">distilgpt2</a></td>
    <td>language-modeling(CLM)</td>
    <td><a href="https://huggingface.co/datasets/wikitext">wikitext</a></td>
    <td>&#10004;</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/distilbert-base-cased">distilbert-base-cased</a></td>
    <td>language-modeling(MLM)</td>
    <td><a href="https://huggingface.co/datasets/wikitext">wikitext</a></td>
    <td>&#10004;</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/Rocketknight1/bert-base-uncased-finetuned-swag">Rocketknight1/bert-base-uncased-finetuned-swag</a></td>
    <td>multiple-choice</td>
    <td><a href="https://huggingface.co/datasets/swag">swag</a></td>
    <td>&#10004;</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/dslim/bert-base-NER">dslim/bert-base-NER</a></td>
    <td>token-classification</td>
    <td><a href="https://huggingface.co/datasets/conll2003">conll2003</a></td>
    <td>&#10004;</td>
  </tr>
</tbody>
</table>



## Length Adaptive Transformers
<table>
<thead>
  <tr>
    <th rowspan="2">Model Name</th>
    <th rowspan="2">Datatype</th>
    <th rowspan="2">Optimization Method</th>
    <th rowspan="2">Modelsize (MB)</th>
    <th colspan="4">Inference Result</th>
  </tr>
  <tr>
    <th>Accuracy(F1)</th>
    <th>Latency(ms)</th>
    <th>GFLOPS**</th>
    <th>Speedup(compared with BERT Base)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>BERT Base</td>
    <td>fp32</td>
    <td>None</td>
    <td>415.47</td>
    <td>88.58</td>
    <td>56.56</td>
    <td>35.3</td>
    <td>1x</td>
  </tr>
  <tr>
    <td>LA-MiniLM</td>
    <td>fp32</td>
    <td>Drop and restore base MiniLMv2</td>
    <td>115.04</td>
    <td>89.28</td>
    <td>16.99</td>
    <td>4.76</td>
    <td>3.33x</td>
  </tr>
  <tr>
    <td>LA-MiniLM(269, 253, 252, 202, 104, 34)*</td>
    <td>fp32</td>
    <td>Evolution search (best config)</td>
    <td>115.04</td>
    <td>87.76</td>
    <td>11.44</td>
    <td>2.49</td>
    <td>4.94x</td>
  </tr>
  <tr>
    <td>QuaLA-MiniLM</td>
    <td>int8</td>
    <td>Quantization base LA-MiniLM</td>
    <td>84.85</td>
    <td>88.85</td>
    <td>7.84</td>
    <td>4.76</td>
    <td>7.21x</td>
  </tr>
  <tr>
    <td>QuaLA-MiniLM(315,251,242,159,142,33)*</td>
    <td>int8</td>
    <td>Evolution search (best config)</td>
    <td>84.86</td>
    <td>87.68</td>
    <td>6.41</td>
    <td>2.55</td>
    <td>8.82x</td>
  </tr>
</tbody>
</table>

>**Note**: * length config apply to Length Adaptive model


>**Note**: ** the multiplication and addition operation amount when model inference  (GFLOPS is obtained from torchprofile tool)


Data is tested on Intel Xeon Platinum 8280 Scalable processor. Configuration detail please refer to [examples](../examples/huggingface/pytorch/question-answering/dynamic/README.md)


## Pruning


<table>
<thead>
  <tr>
    <th>Model</th>
    <th>Task</th>
    <th>Dataset</th>
    <th>Pruning Approach</th>
    <th>Pruning Type</th>
    <th>Framework</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td><a href="https://huggingface.co/distilbert-base-uncased-distilled-squad">distilbert-base-uncased-distilled-squad</a></td>
    <td>question-answering</td>
    <td><a href="https://huggingface.co/datasets/squad">SQuAD</a></td>
    <td>BasicMagnitude</td>
    <td>Unstructured</td>
    <td>Stock PyTorch</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/bert-large-uncased">bert-large-uncased</a></td>
    <td>question-answering</td>
    <td><a href="https://huggingface.co/datasets/squad">SQuAD</a></td>
    <td>Group LASSO</td>
    <td>Structured</td>
    <td>Stock PyTorch</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english">distilbert-base-uncased-finetuned-sst-2-english</a></td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/glue/viewer/sst2/train">SST-2</a></td>
    <td>BasicMagnitude</td>
    <td>Unstructured</td>
    <td>Stock PyTorch/&nbsp;&nbsp; Intel TensorFlow</td>
  </tr>
</tbody>
</table>

## Distillation

### Knowledge Distillation
<table>
<thead>
  <tr>
    <th>Student Model</th>
    <th>Teacher Model</th>
    <th>Task</th>
    <th>Dataset</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td><a href="https://huggingface.co/distilbert-base-uncased">distilbert-base-uncased</a></td>
    <td><a href="https://huggingface.co/textattack/bert-base-uncased-SST-2">bert-base-uncased-SST-2</a></td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/glue/viewer/sst2/train">SST-2</a></td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/distilbert-base-uncased">distilbert-base-uncased</a></td>
    <td><a href="https://huggingface.co/textattack/bert-base-uncased-QNLI">bert-base-uncased-QNLI</a></td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/glue/viewer/qnli/train">QNLI</a></td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/distilbert-base-uncased">distilbert-base-uncased</a></td>
    <td><a href="https://huggingface.co/textattack/bert-base-uncased-QQP">bert-base-uncased-QQP</a></td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/glue/viewer/qqp/train">QQP</a></td>
  </tr>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/distilbert-base-uncased">distilbert-base-uncased</a></td>
    <td><a href="https://huggingface.co/blackbird/bert-base-uncased-MNLI-v1">bert-base-uncased-MNLI-v1</a></td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/glue/viewer/mnli/train">MNLI</a></td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/distilbert-base-uncased">distilbert-base-uncased</a></td>
    <td><a href="https://huggingface.co/csarron/bert-base-uncased-squad-v1">bert-base-uncased-squad-v1</a></td>
    <td>question-answering</td>
    <td><a href="https://huggingface.co/datasets/squad">SQuAD</a></td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D">TinyBERT_General_4L_312D</a></td>
    <td><a href="https://huggingface.co/blackbird/bert-base-uncased-MNLI-v1">bert-base-uncased-MNLI-v1</a></td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/glue/viewer/mnli">MNLI</a></td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/distilroberta-base">distilroberta-base</a></td>
    <td><a href="https://huggingface.co/cointegrated/roberta-large-cola-krishna2020">roberta-large-cola-krishna2020</a></td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/glue/viewer/cola">COLA</a></td>
  </tr>
</tbody>
</table>

### Auto Distillation (NAS Based)
<table>
<thead>
  <tr>
    <th>Model</th>
    <th>Task</th>
    <th>Dataset</th>
    <th>Distillation Teacher</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td><a href="https://huggingface.co/google/mobilebert-uncased">google/mobilebert-uncased</a></td>
    <td>language-modeling(MLM)</td>
    <td><a href="https://huggingface.co/datasets/wikipedia">wikipedia</a></td>
    <td><a href="https://huggingface.co/bert-large-uncased">bert-large-uncased</a></td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/prajjwal1/bert-tiny">prajjwal1/bert-tiny</a></td>
    <td>language-modeling(MLM)</td>
    <td><a href="https://huggingface.co/datasets/wikipedia">wikipedia</a></td>
    <td><a href="https://huggingface.co/bert-base-uncased">bert-base-uncased</a></td>
  </tr>
</tbody>
</table>

## Orchestrate


<table>
<thead>
  <tr>
    <th>Model</th>
    <th>Task</th>
    <th>Dataset</th>
    <th>Distillation Teacher</th>
    <th>Pruning Approch</th>
    <th>Pruning Type</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td rowspan="4"><a href="https://huggingface.co/Intel/distilbert-base-uncased-sparse-90-unstructured-pruneofa">Intel/distilbert-base-uncased-sparse-90-unstructured-pruneofa</a></td>
    <td rowspan="2">question-answering</td>
    <td rowspan="2"><a href="https://huggingface.co/datasets/squad">SQuAD</a></td>
    <td rowspan="2"><a href="https://huggingface.co/distilbert-base-uncased-distilled-squad">distilbert-base-uncased-distilled-squad</a></td>
    <td>PatternLock</td>
    <td>Unstructured</td>
  </tr>
  <tr>
    <td>BasicMagnitude</td>
    <td>Unstructured</td>
  </tr>
  <tr>
    <td rowspan="2">text-classification</td>
    <td rowspan="2"><a href="https://huggingface.co/datasets/glue/viewer/sst2/train">SST-2</a></td>
    <td rowspan="2"><a href="https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english">distilbert-base-uncased-finetuned-sst-2-english</a></td>
    <td>PatternLock</td>
    <td>Unstructured</td>
  </tr>
  <tr>
    <td>BasicMagnitude</td>
    <td>Unstructured</td>
  </tr>
</tbody>
</table>

## Reference Deployment on Neural Engine

### Dense Reference Deployment on Neural Engine
<table>
<thead>
  <tr>
    <th rowspan="2">Model</th>
    <th rowspan="2">Task</th>
    <th rowspan="2">Dataset</th>
    <th colspan="2">Datatype</th>
  </tr>
  <tr>
    <th>INT8</th>
    <th>BF16</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td><a href="https://huggingface.co/distilbert-base-uncased-distilled-squad">bert-large-uncased-whole-word-masking-finetuned-squad</a></td>
    <td>question-answering</td>
    <td><a href="https://huggingface.co/datasets/squad">SQuAD</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion">bhadresh-savani/distilbert-base-uncased-emotion</td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/emotion">emotion</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/textattack/bert-base-uncased-MRPC">textattack/bert-base-uncased-MRPC</td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/glue/viewer/mrpc/train">MRPC</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/textattack/distilbert-base-uncased-MRPC">textattack/distilbert-base-uncased-MRPC</td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/glue/viewer/mrpc/train">MRPC</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/Intel/roberta-base-mrpc">Intel/roberta-base-mrpc</td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/glue/viewer/mrpc/train">MRPC</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/M-FAC/bert-mini-finetuned-mrpc">M-FAC/bert-mini-finetuned-mrpc</td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/glue/viewer/mrpc/train">MRPC</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/gchhablani/bert-base-cased-finetuned-mrpc">gchhablani/bert-base-cased-finetuned-mrpc</td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/glue/viewer/mrpc/train">MRPC</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english">distilbert-base-uncased-finetuned-sst-2-english</td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/glue/viewer/sst2/train">SST-2</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/philschmid/MiniLM-L6-H384-uncased-sst2">philschmid/MiniLM-L6-H384-uncased-sst2</td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/glue/viewer/sst2/train">SST-2</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/moshew/bert-mini-sst2-distilled">moshew/bert-mini-sst2-distilled</td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/glue/viewer/sst2/train">SST-2</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
  </tr>
</tbody>
</table>

### Sparse Reference Deployment on Neural Engine


<table>
<thead>
  <tr>
    <th rowspan="2">Model</th>
    <th rowspan="2">Task</th>
    <th rowspan="2">Dataset</th>
    <th colspan="2">Datatype</th>
  </tr>
  <tr>
    <th>INT8</th>
    <th>BF16</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td><a href="https://huggingface.co/Intel/distilbert-base-uncased-squadv1.1-sparse-80-1x4-block-pruneofa">Intel/distilbert-base-uncased-squadv1.1-sparse-80-1x4-block-pruneofa</td>
    <td>question-answering</td>
    <td><a href="https://huggingface.co/datasets/squad">SQuAD</a></td>
    <td>&#10004;</td>
    <td>WIP :star:</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/Intel/bert-mini-sst2-distilled-sparse-90-1X4-block">Intel/bert-mini-sst2-distilled-sparse-90-1X4-block</td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/glue/viewer/sst2/train">SST-2</a></td>
    <td>&#10004;</td>
    <td>WIP :star:</td>
  </tr>
</tbody>
</table>

