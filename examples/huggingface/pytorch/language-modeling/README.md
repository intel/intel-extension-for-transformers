# Language modeling
There are two types of language modeling, casual and masked. Causal language models are used for text generation and masked language modeling is great for tasks that require a good contextual understanding of an entire squence. In this example, casual/masked language modeling predicts the next/masked token or word in a sequence and return the accuracy validated on dataset. Please see [text-generation](https://github.com/intel/intel-extension-for-transformers/tree/main/examples/huggingface/pytorch/text-generation) if you would like to generate a sentence.

We provide FP32/BF16 inference, INT8 inference, and other advanced compression techniques such as distillation, neural architecture search (NAS) for language modeling task.


<table>
<tbody>
  <tr>
    <td rowspan="4"><a href="https://github.com/intel/intel-extension-for-transformers/tree/main/examples/huggingface/pytorch/language-modeling/distillation" target="_blank" rel="noopener noreferrer">AutoDistillation</a></td>
    <td rowspan="4">Masked language modeling</td>
    <td rowspan="2"><a href="https://github.com/intel/intel-extension-for-transformers/tree/main/examples/huggingface/pytorch/language-modeling/distillation#11-mobilebert" target="_blank" rel="noopener noreferrer">MobileBERT</a></td>
    <td>Teacher model</td>
    <td><a href="https://huggingface.co/bert-large-uncased" target="_blank" rel="noopener noreferrer">bert-large-uncased</a></td>
    <td rowspan="4">train dataset:<br>- English Wikipeadia<br>- BookCorpus</td>
  </tr>
  <tr>
    <td>Student model</td>
    <td><a href="https://huggingface.co/google/mobilebert-uncased" target="_blank" rel="noopener noreferrer">google/mobilebert-uncased</a></td>
  </tr>
  <tr>
    <td rowspan="2"><a href="https://github.com/intel/intel-extension-for-transformers/tree/main/examples/huggingface/pytorch/language-modeling/distillation#12-bert-tiny" target="_blank" rel="noopener noreferrer">BERT-Tiny</a></td>
    <td>Teacher model</td>
    <td><a href="https://huggingface.co/bert-base-uncased" target="_blank" rel="noopener noreferrer">bert-base-uncased</a></td>
  </tr>
  <tr>
    <td>Student model</td>
    <td><a href="https://huggingface.co/bert-base-uncased" target="_blank" rel="noopener noreferrer">prajjwal1/bert-tiny</a></td>
  </tr>
  <tr>
    <td rowspan="4"><a href="https://github.com/intel/intel-extension-for-transformers/tree/main/examples/huggingface/pytorch/language-modeling/nas" target="_blank" rel="noopener noreferrer">Neural architecture search</a></td>
    <td rowspan="4">Masked language modeling</td>
    <td rowspan="2"><a href="https://github.com/intel/intel-extension-for-transformers/tree/main/examples/huggingface/pytorch/language-modeling/nas#11-mobilebert" target="_blank" rel="noopener noreferrer">MobileBERT</a></td>
    <td>Teacher model</td>
    <td><a href="https://huggingface.co/bert-large-uncased" target="_blank" rel="noopener noreferrer">bert-large-uncased</a></td>
    <td rowspan="4">train dataset:<br>- English Wikipeadia<br>- BookCorpus</td>
  </tr>
  <tr>
    <td>Student model</td>
    <td><a href="https://huggingface.co/google/mobilebert-uncased" target="_blank" rel="noopener noreferrer">google/mobilebert-uncased</a></td>
  </tr>
  <tr>
    <td rowspan="2"><a href="https://github.com/intel/intel-extension-for-transformers/tree/main/examples/huggingface/pytorch/language-modeling/nas#12-berttiny" target="_blank" rel="noopener noreferrer">BERT-Tiny</a></td>
    <td>Teacher model</td>
    <td><a href="https://huggingface.co/bert-base-uncased" target="_blank" rel="noopener noreferrer">bert-base-uncased</a></td>
  </tr>
  <tr>
    <td>Student model</td>
    <td><a href="https://huggingface.co/prajjwal1/bert-tiny" target="_blank" rel="noopener noreferrer">prajjwal1/bert-tiny</a></td>
  </tr>
  <tr>
    <td rowspan="4"><a href="https://github.com/intel/intel-extension-for-transformers/tree/main/examples/huggingface/pytorch/language-modeling/inference" target="_blank" rel="noopener noreferrer">Inference</a></td>
    <td rowspan="4">Causal language modeling</td>
    <td rowspan="2">Float32</td>
    <td>Imperative mode</td>
    <td rowspan="4">- <a href="https://huggingface.co/EleutherAI/gpt-j-6b" target="_blank" rel="noopener noreferrer"><b>EleutherAI/gpt-j-6b</b></a><br>- <a href="https://huggingface.co/facebook/opt-1.3b" target="_blank" rel="noopener noreferrer"><b>facebook/opt-1.3b</b></a><br>- <a href="https://huggingface.co/facebook/opt-2.7b" target="_blank" rel="noopener noreferrer"><b>facebook/opt-2.7b</b></a><br>- <a href="https://huggingface.co/facebook/opt-6.7b" target="_blank" rel="noopener noreferrer"><b>facebook/opt-6.7b</b></a><br>- <a href="https://huggingface.co/decapoda-research/llama-7b-hf" target="_blank" rel="noopener noreferrer"><b>decapoda-research/llama-7b-hf</b></a><br>- <a href="https://huggingface.co/decapoda-research/llama-13b-hf" target="_blank" rel="noopener noreferrer"><b>decapoda-research/llama-13b-hf</b></a></td>
    <td rowspan="4">validation dataset<br>- lambada_openai<br>- lambada_standard<br>- piqa<br>- winogrande<br>- copa<br>- hellaswag<br>- openbookqa</td>
  </tr>
  <tr>
    <td>TorchScript mode</td>
  </tr>
  <tr>
    <td rowspan="2">Bfloat16</td>
    <td>Imperative mode</td>
  </tr>
  <tr>
    <td>TorchScript mode</td>
  </tr>
  <tr>
    <td rowspan="13"><a href="https://github.com/intel/intel-extension-for-transformers/tree/main/examples/huggingface/pytorch/language-modeling/quantization" target="_blank" rel="noopener noreferrer">Qunatization</a></td>
    <td rowspan="7">Causal language modeling</td>
    <td rowspan="4">Post training static quantization</td>
    <td>TorchScript mode</td>
    <td>- <a href="https://huggingface.co/EleutherAI/gpt-j-6b" target="_blank" rel="noopener noreferrer"><b>EleutherAI/gpt-j-6b<b/></a><br>- <a href="https://huggingface.co/facebook/opt-1.3b" target="_blank" rel="noopener noreferrer"><b>facebook/opt-1.3b</b></a><br>- <a href="https://huggingface.co/facebook/opt-2.7b" target="_blank" rel="noopener noreferrer"><b>facebook/opt-2.7b</b></a><br>- <a href="https://huggingface.co/facebook/opt-6.7b" target="_blank" rel="noopener noreferrer"><b>facebook/opt-6.7b</b></a><br>- <a href="https://huggingface.co/decapoda-research/llama-7b-hf" target="_blank" rel="noopener noreferrer"><b>decapoda-research/llama-7b-hf</b></a><br>- <a href="https://huggingface.co/decapoda-research/llama-13b-hf" target="_blank" rel="noopener noreferrer">decapoda-research/llama-13b-hf</a></td>
    <td>calibration dataset<br>- NeelNanda/pile-10k<br>validation dataset<br>- lambada_openai<br>- lambada_standard<br>- piqa<br>- hellaswag<br>- winogrande</td>
  </tr>
  <tr>
    <td rowspan="3">FXGraph mode</td>
    <td><a href="https://huggingface.co/EleutherAI/gpt-j-6b" target="_blank" rel="noopener noreferrer"><b>EleutherAI/gpt-j-6b</b></a><br><a href="https://huggingface.co/EleutherAI/gpt-neo-125m" target="_blank" rel="noopener noreferrer"><b>EleutherAI/gpt-neo-125m</b></a></td>
    <td>calibration dataset<br>- wikitext<br>validation dataset<br>- wikitext</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/bigscience/bloom-560m" target="_blank" rel="noopener noreferrer"><b>bigscience/bloom-560m</b></a></td>
    <td>calibration dataset<br>- lambada<br>validation dataset<br>- lambada</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/abeja/gpt-neox-japanese-2.7b" target="_blank" rel="noopener noreferrer"><b>abeja/gpt-neox-japanese-2.7b</b></a></td>
    <td>calibration dataset<br>- oscar<br>validation dataset<br>- oscar</td>
  </tr>
  <tr>
    <td rowspan="2">Post training dynamic quantization</td>
    <td rowspan="2">FXGraph mode</td>
    <td><a href="https://huggingface.co/EleutherAI/gpt-j-6b" target="_blank" rel="noopener noreferrer"><b>EleutherAI/gpt-j-6b</b></a><br><a href="https://huggingface.co/EleutherAI/gpt-neox-125m" target="_blank" rel="noopener noreferrer"><b>EleutherAI/gpt-neox-125m</b></a></td>
    <td>validation dataset<br>- wikitext</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/abeja/gpt-neox-japanese-2.7b" target="_blank" rel="noopener noreferrer"><b>abeja/gpt-neox-japanese-2.7b</b></a></td>
    <td>validation dataset<br>- oscar</td>
  </tr>
  <tr>
    <td>Quant aware training</td>
    <td>FxGraph mode</td>
    <td><a href="https://huggingface.co/abeja/gpt-neox-japanese-2.7b" target="_blank" rel="noopener noreferrer"><b>abeja/gpt-neox-japanese-2.7b</b></a></td>
    <td>train dataset<br>- wikitext<br>validation dataset<br>- wikitext</td>
  </tr>
  <tr>
    <td rowspan="3">Masked language modeling</td>
    <td>Post training static quantization</td>
    <td rowspan="3">FXGraph mode</td>
    <td><a href="https://huggingface.co/bert-base-uncased" target="_blank" rel="noopener noreferrer">bert-base-uncased</a></td>
    <td>validation dataset<br>- wikitext</td>
  </tr>
  <tr>
    <td>Post training dynamic quantization</td>
    <td><a href="https://huggingface.co/bert-base-uncased" target="_blank" rel="noopener noreferrer">bert-base-uncased</a></td>
    <td>validation dataset<br>- wikitext</td>
  </tr>
  <tr>
    <td>Quant aware training</td>
    <td><a href="https://huggingface.co/bert-base-uncased" target="_blank" rel="noopener noreferrer">bert-base-uncased</a></td>
    <td>validation dataset<br>- wikitext</td>
  </tr>
  <tr>
    <td rowspan="3">Permutation language modeling</td>
    <td>Post training static quantization</td>
    <td rowspan="3">FXGraph mode</td>
    <td><a href="https://huggingface.co/xlnet-base-cased" target="_blank" rel="noopener noreferrer">xlnet-base-cased</a></td>
    <td>calibration dataset<br>- wikitext<br>validatuion dataset<br>- wikitext</td>
  </tr>
  <tr>
    <td>Post training dynamic quantization</td>
    <td><a href="https://huggingface.co/xlnet-base-cased" target="_blank" rel="noopener noreferrer">xlnet-base-cased</a></td>
    <td>validation dataset<br>- wikitext</td>
  </tr>
  <tr>
    <td>Quant aware training</td>
    <td><a href="https://huggingface.co/xlnet-base-cased" target="_blank" rel="noopener noreferrer">xlnet-base-cased</a></td>
    <td>train dataset<br>- wikitext<br>validatuion dataset<br>- wikitext</td>
  </tr>
</tbody>
</table>
