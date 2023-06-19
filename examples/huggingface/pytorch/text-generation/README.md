# Text Generation
Text generation is a common task in natural language processing field. It leverages knowledge in computational linguistics and artificial intelligence to automatically generate natural language texts, which can satisfy certain communicative requirements.

We provide [quantizatioin script](./quantization/run_generation.py) for [EleutherAI/gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6B),  [decapoda-research/llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf), [decapoda-research/llama-13b-hf](https://huggingface.co/decapoda-research/llama-13b-hf), [databricks/dolly-v2-3b](https://huggingface.co/databricks/dolly-v2-3b), [bigscience/bloom-7b1](https://huggingface.co/bigscience/bloom-7b1), [facebook/opt-1.3b](https://huggingface.co/facebook/opt-1.3b), [facebook/opt-2.7b](https://huggingface.co/facebook/opt-2.7b) and [facebook/opt-6.7b](https://huggingface.co/facebook/opt-6.7b) .


We also provide Neural Engine [inference script](./deployment/run_llm.py) for [EleutherAI/gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6B) and [decapoda-research/llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf)


## Validated Models


### Latency (ms)


Neural Engine: 1.0.1

Batch Size: 1

Input Length: 32

Output Length: 32

| Model |  Beam  | FP32 | BF16 | INT8 |
|---------------------|:------:|:----------------------:|-----------------------|-----------------------------------|
| [EleutherAI/gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6B) | 4 | 4163.67 (ms) | 1879.61 (ms) | 1612.24 (ms) |
| [decapoda-research/llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf) | 4 | 4356.79 (ms) | 1941.63 (ms) | 1773.22 (ms) |

> Note: Performance results test on ​​06/09/2023 with Intel(R) Xeon(R) Platinum 8480+.
Performance varies by use, configuration and other factors. See platform configuration for configuration details. For more complete information about performance and benchmark results, visit www.intel.com/benchmarks


### Platform Configuration


<table>
<tbody>
  <tr>
    <td>Manufacturer</td>
    <td>Quanta Cloud Technology Inc</td>
  </tr>
  <tr>
    <td>Product Name</td>
    <td>QuantaGrid D54Q-2U</td>
  </tr>
  <tr>
    <td>OS</td>
    <td>CentOS Stream 8</td>
  </tr>
  <tr>
    <td>BIOS Version</td>
    <td>3A14.TEL2P1</td>
  </tr>
  <tr>
    <td>Kernel</td>
    <td>5.16.0-rc1-intel-next-00543-g5867b0a2a125</td>
  </tr>
  <tr>
    <td>Microcode</td>
    <td>0x2b0001b0</td>
  </tr>
  <tr>
    <td>IRQ Balance</td>
    <td>Eabled</td>
  </tr>
  <tr>
    <td>CPU Model</td>
    <td>Intel(R) Xeon(R) Platinum 8480+</td>
  </tr>
  <tr>
    <td>Base Frequency</td>
    <td>2.0GHz</td>
  </tr>
  <tr>
    <td>Maximum Frequency</td>
    <td>3.8GHz</td>
  </tr>
  <tr>
    <td>All-core Maximum Frequency</td>
    <td>3.0GHz</td>
  </tr>
  <tr>
    <td>CPU(s)</td>
    <td>224</td>
  </tr>
  <tr>
    <td>Thread(s) per Core</td>
    <td>2</td>
  </tr>
  <tr>
    <td>Core(s) per Socket</td>
    <td>56</td>
  </tr>
  <tr>
    <td>Socket(s)</td>
    <td>2</td>
  </tr>
  <tr>
    <td>NUMA Node(s)</td>
    <td>2</td>
  </tr>
  <tr>
    <td>Turbo</td>
    <td>Enabled</td>
  </tr>
  <tr>
    <td>FrequencyGoverner</td>
    <td>Performance</td>
  </tr>
</tbody>
</table>





### Last Word Accuracy with Smoothquant


Neural Compressor: 2.1

IPEX (Intel Extension for PyTorch): 2.1

Dataset: lambada-openai


| Model |  Smoothquant Config  | FP32  | BF16 | INT8 (mixed precision) |
|---------------------|:------:|:----------------------:|-----------------------|-----------------------------------|
| EleutherAI/gpt-j-6B | alpha 1.0 | 68.31% | 67.86% | 68.21% (w/o BF16) |
| decapoda-research/llama-7b-hf | alpha 0.8 | 73.61% | 73.26% | 73.57% (w/o BF16) |
| decapoda-research/llama-13b-hf | alpha 0.7 | 76.27% | 76.01% | 75.90% (w/o BF16) |
| decapoda-research/llama-30b-hf | alpha 0.7 | 77.57% | 77.53% | 78.40% (w/o BF16) |
| facebook/opt-125m   | alpha 0.5 | 37.9% | 37.63% | 37.57% (w/o BF16) |
| facebook/opt-350m   | alpha 0.8 | 45.16% | 45.06% | 45.53% (w/o BF16) |
| facebook/opt-2.7b   | alpha 0.5 | 63.65% | 63.23% | 64.04% (w/ BF16) |
| facebook/opt-6.7b   | alpha 0.5 | 67.69% | 67.36% | 68.04% (w/ BF16) |
| facebook/opt-13b   | alpha 0.5 | 68.72% | 67.84% | 68.14% (w/o BF16) |
| facebook/opt-30b   | alpha 0.5 | 71.49% | 70.87% | 71.28% (w/o BF16) |
| bigscience/bloom-560m   | alpha 0.5 | 35.4% | 25.56% | 35.36% (w/o BF16) |
| bigscience/bloom-1b7   | alpha 0.5 | 46.34% | 45.7% | 49.06% (w/ BF16) |
| bigscience/bloom-3b   | alpha 0.8 | 51.8% | 51.35% | 51.85% (w/o BF16) |
| bigscience/bloom-7b1   | alpha 0.5 | 57.64% | 57.23% | 59.77% (w/ BF16) |
| databricks/dolly-v1-6b   | alpha 0.8 | 68.66% | 67.96% | 68.95% (w/o BF16) |
| databricks/dolly-v2-3b   | alpha 0.5 | 62.97% | 60.86% | 62.47% (w/o BF16) |


