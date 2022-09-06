Examples 
===
NLP Toolkit is a powerful toolkit with multiple model optimization techniques for Natural Language Processing Models, including quantization, pruning, distillation, auto distillation and orchestrate. Meanwhile NLP Toolkit provides Neural Engine, an optimized backend for NLP models to demonstrate the deployment.

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
    <th>QuantizationAwareTraining</th>
    <th>No Trainer quantization</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td><a href="https://huggingface.co/bert-base-uncased">bert-base-uncased</a></td>
    <td>language-modeling(MLM)</td>
    <td><a href="https://huggingface.co/datasets/wikitext">wikitext</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
    <td>WIP :star:</td>
    <td>WIP :star:</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/xlnet-base-case">xlnet-base-cased</a></td>
    <td>language-modeling(PLM)</td>
    <td><a href="https://huggingface.co/datasets/wikitext">wikitext</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
    <td>WIP :star:</td>
    <td>WIP :star:</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/EleutherAI/gpt-neo-125M">EleutherAI/gpt-neo-125M</a></td>
    <td>language-modeling(CLM)</td>
    <td><a href="https://huggingface.co/datasets/wikitext">wikitext</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
    <td>WIP :star:</td>
    <td>WIP :star:</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/sshleifer/tiny-ctrl">sshleifer/tiny-ctrl</a></td>
    <td>language-modeling(CLM)</td>
    <td><a href="https://huggingface.co/datasets/wikitext">wikitext</a></td>
    <td>WIP :star:</td>
    <td>&#10004;</td>
    <td>WIP :star:</td>
    <td>WIP :star:</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/ehdwns1516/bert-base-uncased_SWAG">ehdwns1516/bert-base-uncased_SWAG</a></td>
    <td>multiple-choice</td>
    <td><a href="https://huggingface.co/datasets/swag">swag</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
    <td>WIP :star:</td>
    <td>WIP :star:</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/distilbert-base-uncased-distilled-squad">distilbert-base-uncased-distilled-squad</a></td>
    <td>question-answering</td>
    <td><a href="https://huggingface.co/datasets/squad">SQuAD</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
    <td>WIP :star:</td>
    <td>WIP :star:</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/lvwerra/pegasus-samsum">lvwerra/pegasus-samsum</a></td>
    <td>summarization</td>
    <td><a href="https://huggingface.co/datasets/samsum">samsum</a></td>
    <td>&#10004;</td>
    <td>WIP :star:</td>
    <td>WIP :star:</td>
    <td>WIP :star:</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/textattack/bert-base-uncased-MRPC">textattack/bert-base-uncased-MRPC</a></td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/glue/viewer/mrpc/train">MRPC</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
    <td>&#10004;</td>
    <td>WIP :star:</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/echarlaix/bert-base-uncased-sst2-acc91.1-d37-hybrid">echarlaix/bert-base-uncased-sst2-acc91.1-d37-hybrid</a></td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/glue/viewer/sst2/train">SST-2</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
    <td>WIP :star:</td>
    <td>&#10004;</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english">distilbert-base-uncased-finetuned-sst-2-english</a></td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/glue/viewer/sst2/train">SST-2</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
    <td>WIP :star:</td>
    <td>WIP :star:</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/elastic/distilbert-base-uncased-finetuned-conll03-english">elastic/distilbert-base-uncased-finetuned-conll03-english</a></td>
    <td>token-classification</td>
    <td><a href="https://huggingface.co/datasets/conll2003">conll2003</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
    <td>WIP :star:</td>
    <td>WIP :star:</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/t5-small">t5-small</a></td>
    <td>translation</td>
    <td><a href="https://huggingface.co/datasets/wmt16">wmt16</a></td>
    <td>&#10004;</td>
    <td>WIP :star:</td>
    <td>WIP :star:</td>
    <td>WIP :star:</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/Helsinki-NLP/opus-mt-en-ro">Helsinki-NLP/opus-mt-en-ro</a></td>
    <td>translation</td>
    <td><a href="https://huggingface.co/datasets/wmt16">wmt16</a></td>
    <td>&#10004;</td>
    <td>WIP :star:</td>
    <td>WIP :star:</td>
    <td>WIP :star:</td>
  </tr>
</tbody>
</table>


### IPEX examples
<table>
<thead>
  <tr>
    <th>Model</th>
    <th>Task</th>
    <th>Dataset</th>
    <th>PostTrainingDynamic</th>
    <th>PostTrainingStatic</th>
    <th>QuantizationAwareTraining</th>
    <th>No Trainer quantization</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td><a href="https://huggingface.co/distilbert-base-uncased-distilled-squad">distilbert-base-uncased-distilled-squad</a></td>
    <td>question-answering</td>
    <td><a href="https://huggingface.co/datasets/squad">SQuAD</a></td>
    <td>WIP :star:</td>
    <td>&#10004;</td>
    <td>WIP :star:</td>
    <td>WIP :star:</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/bert-large-uncased-whole-word-maskinuned-squad">bert-large-uncased-whole-word-maskinuned-squad</a></td>
    <td>question-answering</td>
    <td><a href="https://huggingface.co/datasets/squad">SQuAD</a></td>
    <td>WIP :star:</td>
    <td>&#10004;</td>
    <td>WIP :star:</td>
    <td>WIP :star:</td>
  </tr>
</tbody>
</table>

### Intel TensorFlow examples
<table>
<thead>
  <tr>
    <th>Model</th>
    <th>Task</th>
    <th>Dataset</th>
    <th>PostTrainingDynamic</th>
    <th>PostTrainingStatic</th>
    <th>QuantizationAwareTraining</th>
    <th>No Trainer quantization</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td><a href="https://huggingface.co/bert-base-cased-finetuned-mrpc">bert-base-cased-finetuned-mrpc</a></td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/glue/viewer/mrpc/train">MRPC</a></td>
    <td>WIP :star:</td>
    <td>&#10004;</td>
    <td>WIP :star:</td>
    <td>WIP :star:</td>
  </tr>
</tbody>
</table>


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
    <td>Stock PyTorch/<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Intel TensorFlow</td>
  </tr>
</tbody>
</table>

## Distillation

### Knowledge Distillation
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
    <td rowspan="2"><a href="https://huggingface.co/Intel/distilbert-base-uncased-sparse-90-unstructured-pruneofa">Intel/distilbert-base-uncased-sparse-90-unstructured-pruneofa</a></td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/glue/viewer/sst2/train">SST-2</a></td>
    <td><a href="https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english">distilbert-base-uncased-finetuned-sst-2-english</a></td>
  </tr>
  <tr>
    <td>question-answering</td>
    <td><a href="https://huggingface.co/datasets/squad">SQuAD</a></td>
    <td><a href="https://huggingface.co/distilbert-base-uncased-distilled-squad">distilbert-base-uncased-distilled-squad</a></td>
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


<table>
<thead>
  <tr>
    <th rowspan="2">Model</th>
    <th rowspan="2">Task</th>
    <th rowspan="2">Dataset</th>
    <th colspan="2">Dense</th>
    <th colspan="2">Sparse</th>
  </tr>
  <tr>
    <th>INT8</th>
    <th>BF16</th>
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
    <td>WIP :star:</td>
    <td>WIP :star:</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion">bhadresh-savani/distilbert-base-uncased-emotion</td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/emotion">emotion</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
    <td>WIP :star:</td>
    <td>WIP :star:</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/textattack/bert-base-uncased-MRPC">textattack/bert-base-uncased-MRPC</td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/glue/viewer/mrpc/train">MRPC</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
    <td>WIP :star:</td>
    <td>WIP :star:</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/textattack/distilbert-base-uncased-MRPC">textattack/distilbert-base-uncased-MRPC</td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/glue/viewer/mrpc/train">MRPC</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
    <td>WIP :star:</td>
    <td>WIP :star:</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/Intel/roberta-base-mrpc">Intel/roberta-base-mrpc</td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/glue/viewer/mrpc/train">MRPC</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
    <td>WIP :star:</td>
    <td>WIP :star:</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/M-FAC/bert-mini-finetuned-mrpc">M-FAC/bert-mini-finetuned-mrpc</td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/glue/viewer/mrpc/train">MRPC</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
    <td>WIP :star:</td>
    <td>WIP :star:</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/gchhablani/bert-base-cased-finetuned-mrpc">gchhablani/bert-base-cased-finetuned-mrpc</td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/glue/viewer/mrpc/train">MRPC</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
    <td>WIP :star:</td>
    <td>WIP :star:</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english">distilbert-base-uncased-finetuned-sst-2-english</td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/glue/viewer/sst2/train">SST-2</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
    <td>WIP :star:</td>
    <td>WIP :star:</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/philschmid/MiniLM-L6-H384-uncased-sst2">philschmid/MiniLM-L6-H384-uncased-sst2</td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/glue/viewer/sst2/train">SST-2</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
    <td>WIP :star:</td>
    <td>WIP :star:</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/moshew/bert-mini-sst2-distilled">moshew/bert-mini-sst2-distilled</td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/glue/viewer/sst2/train">SST-2</a></td>
    <td>&#10004;</td>
    <td>&#10004;</td>
    <td>WIP :star:</td>
    <td>WIP :star:</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/Intel/bert-mini-sst2-distilled-sparse-90-1X4-block">Intel/bert-mini-sst2-distilled-sparse-90-1X4-block</td>
    <td>text-classification</td>
    <td><a href="https://huggingface.co/datasets/glue/viewer/sst2/train">SST-2</a></td>
    <td>N/A</td>
    <td>N/A</td>
    <td>&#10004;</td>
    <td>WIP :star:</td>
  </tr>
</tbody>
</table>
