Validated Model Performance
============
1. [LLM Quantization](#llm-quantization)

2. [LLM Runtime Inference based on Pytorch Mode](#llm-runtime-inference-based-on-pytorch-mode)

    2.1 [LLMs](#llms)

    2.2 [Stable Diffusion](#stable-diffusion)

    2.3 [Electra](#electra)

3. [LLM Runtime (GGML-Compatible)](#llm-runtime-GGML-compatible)

    3.1 [MPT-7B](#mpt-7b)

    3.2 [GPT-j-6B](#gpt-j-6b)

    3.3 [Falcon-7B](#falcon-7b)

    3.4 [GPT-NEOX-20B](#gpt-neox-20b)

    3.5 [Dolly-V2-3B](#dolly-v2-3b)

    3.6 [OPT-1.3B](#opt-13b)

    3.7 [StarCoder-3B](#starcoder-3b)

4. [LLM Finetuning](#llm-finetuning)


System summary: Test by Intel on 09/19/2023. 1-node, 1x Intel(R) Xeon(R) Platinum 8480+ @3.8GHz, 56 cores/socket, HT On, Turbo On, Total Memory 256GB (16x16GB DDR5 4800 MT/s [4800 MT/s]), BIOS 3A14.TEL2P1, microcode 0x2b0001b0,
CentOS Stream 8, gcc (GCC) 8.5.0 20210514 (Red Hat 8.5.0-10), DL Models, Frameworks/Backends: PyTorch/ONNXRT/[LLM Runtime](../intel_extension_for_transformers/llm/runtime/)/GGML, Datatype: FP32/INT8/BF16/FP8.
Using 1 socket, 56 cores/instance, 1 instance and batch size 1

Performance varies by use, configuration and other factors.
For more complete information about performance and benchmark results, visit [www.intel.com/benchmarks](https://edc.intel.com/content/www/us/en/products/performance/benchmarks/overview/)

## LLM Quantization
Environment:

Pytorch: 2.0.1+cpu

Intel Extension for Pytorch: 2.0.100+cpu

Intel Neural Compressor: 2.3


|           |            |           | INT8        |           | FP32        |          |  INT8/FP32      |             |
| --------- | ---------- | ----------| ----------- | --------- | ----------- | -------- | --------------- | ----------- |
| Framework | Model      | Datasets  | Throughput (samples/sec) | Accuracy | Throughput (samples/sec) | Accuracy | Throughput Gain | Relative Accuracy (INT8- FP32)/FP32 |
| pytorch   | opt_1.3b      | NeelNanda/pile-10k | 34.07                    | 57.05%            | 19.92                    | 57.89% | 1.71 | \-1.44% |
| pytorch   | bloom_1b7     | NeelNanda/pile-10k | 29.74                    | 49.95%            | 13.3                     | 46.34% | 2.24 | 7.79% |
| pytorch   | bloom_7b1     | NeelNanda/pile-10k | 12.36                    | 60.14%            | 3.22                     | 57.64% | 3.83 | 4.34% |
| pytorch   | opt_2.7b      | NeelNanda/pile-10k | 23.19                    | 63.67%            | 12.24                    | 63.65% | 1.89 | 0.03% |
| pytorch   | opt_6.7b      | NeelNanda/pile-10k | 13.5                     | 67.01%            | 4.1                      | 67.69% | 3.29 | \-1.00% |
| pytorch   | gpt_j_6b      | NeelNanda/pile-10k | 10.76                    | 67.59%            | 4.38                     | 68.31% | 2.46 | \-1.05% |
| pytorch   | flan_t5_large | samsum             | 69.75                    | 46.25 (rougeLsum) | 33.16                    | 47.67 (rougeLsum) | 2.1 | \-2.99% |
| pytorch   | gpt_neox_clm  | wikitext           | 1.47                     | 4.04 (eval_loss)  | 0.65                     | 3.52 (eval_loss) | 2.27 | \-14.78% |
| pytorch   | gpt_j_6b_clm  | wikitext           | 0.86                     | 3 (eval_loss)     | 0.28                     | 2.34 (eval_loss) | 3.1 | \-28.67% |
| onnx      | whisper_large | lambda-openai      | 2.11                     | 97.07%            | 1.13                     | 96.96% | 1.87 | 0.12% |



## LLM Runtime Inference based on Pytorch Mode
Environment:

Pytorch: 2.0.1+cpu

### LLMs


| Framework | Model                 | Input    | Output |  INT8      | FP32      | BF16      | FP8       | INT8/FP32 | BF16/FP32 | FP8/FP32 |
| --------- | --------------------- | -------- | ------ |  --------- | --------- | --------- | --------- | --------- | --------- | -------- |
| pytorch   | gpt-neox-20b          | 32       | 32  | 9283 (ms) |           |           |           |           |           |          |
| pytorch   | dolly-v2-3b           | 32       | 32  | 3191 (ms) | 3798 (ms) | 2689 (ms) | 1.19x     | 1.41x     |           |
| pytorch   | gpt-j-6b-pruned       | 32       | 32  | 4523 (ms) | 2421 (ms) | 1758 (ms) | 1.87x     | 2.57x     |
| pytorch   | gpt-j-6b              | 32       | 32  | 1658 (ms) | 4561 (ms) | 2429 (ms) | 1793 (ms) | 2.75x     | 1.88x     | 2.54x    |



### Stable Diffusion


| Model                 | Steps | Output   | INT8                | FP32      | BF16                  | INT8+BF16\*           | INT8/FP32 | BF16/FP32 |
| --------------------- | ----- | -------- | ------------------- | --------- | --------------------- | --------------------- | --------- | --------- |
| stable_diffusion_v2_1 | 20    | 512\*512 |                     | 16.98 (s) | 2.83 (s)              |                       |           | 6.00x     |
| stable_diffusion_v1_5 | 20    | 512\*512 | 2.18 (s)            | 10.94 (s) | 2.74 (s)              |                       | 5.01x     | 3.99x     |
| stable_diffusion_v1_5 | 50    | 512\*512 | 5.2 (s) / FID=35.46 |           | 6.3 (s) / FID = 31.07 | 5.5 (s) / FID = 30.58 |           |           |
| stable_diffusion_v1_4 | 20    | 512\*512 |                     | 11.39 (s) | 2.83 (s)              |                       |           | 4.02x     |


>**Note**: *Only works when steps = 50, using BF16 for inference from steps 1 to 5 and from steps 46 to 50, and INT8 for inference from steps 6 to 45. In this inference mode, accuracy and speed can achieve a good balance.


### Electra


|                                    |            |            | FP32         | BF16         | BF16/FP32 |
| ---------------------------------- | ---------- | ---------- | ------------ | ------------ | --------- |
| Model                              | Batch Size | Seq Length | Latency (ms) | Latency (ms) | Latency   |
| electra_base_chinese_discriminator | 1          | 16         | 11.50        | 4.30         | 2.67x     |
| |4                                  | 16         | 5.50       | 1.80         | 3.06x        |
| |8                                  | 16         | 6.20       | 1.70         | 3.65x        |
| |16                                 | 16         | 5.60       | 1.30         | 4.31x        |
| |32                                 | 16         | 5.70       | 1.20         | 4.75x        |
| |64                                 | 16         | 5.20       | 1.10         | 4.73x        |
| electra_base_chinese_generator     | 1          | 128        | 13.72        | 3.89         | 3.53x     |
| |4                                  | 128        | 11.60      | 2.83         | 4.10x        |
| |8                                  | 128        | 11.44      | 2.85         | 4.01x        |
| |16                                 | 128        | 12.04      | 2.70         | 4.46x        |
| |32                                 | 128        | 11.29      | 2.52         | 4.48x        |
| |64                                 | 128        | 11.75      | 2.54         | 4.63x        |



## LLM Runtime (GGML-Compatible)
Environment:
GCC / G++:  12.1.0
Transformers version: 4.35.2 


### MPT-7B

| Backend    | Input | Output | Cores/Instance | Precision | Compute Type | Group Size | Next Token(ms) | Memory mean used (Top 50%) MB | First Token(ms) | Total Latency(ms) | P90 Latency(ms) | P99 Latency(ms) |
| ---------- | ----- | ------ | -------------- | --------- | ------------ | ---------- | -------------- | ----------------------------- | --------------- | ----------------- | --------------- | --------------- |
| LLM Runtime | 32    | 32     | 32             | INT4      | INT8         | 128        | 36.95          | 3522                          | 108.74          | 958.5             | 37.24           | 92.32           |
| LLM Runtime | 1024  | 32     | 32             | INT4      | INT8         | 128        | 46.69          | 4913                          | 15834           | 17281             | 46.83           | 10940           |
| LLM Runtime | 32    | 32     | 48             | INT4      | INT8         | 128        | 34.76          | 5206                          | 100.94          | 900               | 34.9            | 85.92           |
| LLM Runtime | 1024  | 32     | 48             | INT4      | INT8         | 128        | 44.98          | 5147                          | 15506           | 16901             | 45.38           | 10713           |
| LLM Runtime | 32    | 32     | 56             | INT4      | INT8         | 128        | 35.84          | 5230                          | 98.71           | 922               | 36.07           | 84.33           |
| LLM Runtime | 1024  | 32     | 56             | INT4      | INT8         | 128        | 45.54          | 5197                          | 15180           | 16591             | 45.73           | 10488           |
| LLM Runtime | 32    | 32     | 32             | INT4      | INT8         | 32         | 38.33          | 4101                          | 157.31          | 1345              | 38.59           | 120.53          |
| LLM Runtime | 1024  | 32     | 32             | INT4      | INT8         | 32         | 48.19          | 5346                          | 17178           | 18672             | 48.35           | 11868           |
| LLM Runtime | 32    | 32     | 48             | INT4      | INT8         | 32         | 37.75          | 5199                          | 140.79          | 1310              | 37.94           | 108.99          |
| LLM Runtime | 1024  | 32     | 48             | INT4      | INT8         | 32         | 47.21          | 5282                          | 17245           | 18708             | 47.36           | 11914           |
| LLM Runtime | 32    | 32     | 56             | INT4      | INT8         | 32         | 38.04          | 5227                          | 137.21          | 1316              | 38.19           | 106.53          |
| LLM Runtime | 1024  | 32     | 56             | INT4      | INT8         | 32         | 47.88          | 5274                          | 17454           | 18939             | 48.15           | 12058           |
| GGML       | 32    | 32     | 32             | INT4      | INT8         | 32         | 37.92          | 4047                          | 447.6           | 1622              | 38.26           | 320.8           |
| GGML       | 1024  | 32     | 32             | INT4      | INT8         | 32         | 47.74          | 5207                          | 26552           | 28032             | 48.03           | 18336           |
| GGML       | 32    | 32     | 48             | INT4      | INT8         | 32         | 34.78          | 5192                          | 330.06          | 1408              | 35.02           | 238.66          |
| GGML       | 1024  | 32     | 48             | INT4      | INT8         | 32         | 44.64          | 5231                          | 22389           | 23772             | 44.81           | 15462           |
| GGML       | 32    | 32     | 56             | INT4      | INT8         | 32         | 34.53          | 5225                          | 313.45          | 1383              | 34.79           | 227.08          |
| GGML       | 1024  | 32     | 56             | INT4      | INT8         | 32         | 44.64          | 5242                          | 21568           | 22951             | 44.86           | 14896           |


### GPT-j-6B

| Backend    | Input | Output | Cores/Instance | Precision | Compute Type | Group Size | Next Token(ms) | Memory mean used (Top 50%) MB | First Token(ms) | Total Latency(ms) | P90 Latency(ms) | P99 Latency(ms) |
| ---------- | ----- | ------ | -------------- | --------- | ------------ | ---------- | -------------- | ----------------------------- | --------------- | ----------------- | --------------- | --------------- |
| LLM Runtime | 32    | 32     | 32             | INT4      | INT8         | 128        | 23.59          |  4018                         | 62.48           | 793.86            | 23.82           | 50.55           |
| LLM Runtime | 1024  | 32     | 32             | INT4      | INT8         | 128        | 26.2           | 4036                          | 2055            | 2867              | 26.43           | 1426            |
| LLM Runtime | 2012  | 32     | 32             | INT4      | INT8         | 128        | 29.21          | 4553                          | 6114            | 7019              | 29.33           | 4228            |
| LLM Runtime | 32    | 32     | 48             | INT4      | INT8         | 128        | 21.56          | 5230                          | 60.56           | 729               | 21.75           | 48.68           |
| LLM Runtime | 1024  | 32     | 48             | INT4      | INT8         | 128        | 23.92          | 5212                          | 1763            | 2504              | 24.17           | 1224            |
| LLM Runtime | 2012  | 32     | 48             | INT4      | INT8         | 128        | 26.62          | 5119                          | 5230            | 6055              | 26.81           | 3617            |
| LLM Runtime | 32    | 32     | 56             | INT4      | INT8         | 128        | 21.98          | 5244                          | 60.85           | 742.08            | 22.28           | 49.05           |
| LLM Runtime | 1024  | 32     | 56             | INT4      | INT8         | 128        | 24.54          | 5234                          | 2007            | 2768              | 24.7            | 1393            |
| LLM Runtime | 2012  | 32     | 56             | INT4      | INT8         | 32         | 27.16          | 5184                          | 5151            | 5993              | 27.4            | 3563            |
| LLM Runtime | 32    | 32     | 32             | INT4      | INT8         | 32         | 25.35          | 3739                          | 107.52          | 893.52            | 25.42           | 82.17           |
| LLM Runtime | 1024  | 32     | 32             | INT4      | INT8         | 32         | 28.04          | 4435                          | 3405            | 4275.2            | 28.07           | 2359            |
| LLM Runtime | 2012  | 32     | 32             | INT4      | INT8         | 32         | 30.36          | 4914                          | 8916            | 9857              | 30.42           | 6161            |
| LLM Runtime | 32    | 32     | 48             | INT4      | INT8         | 32         | 24.09          | 5228                          | 95.24           | 842.1             | 24.13           | 74.4            |
| LLM Runtime | 1024  | 32     | 48             | INT4      | INT8         | 32         | 26.65          | 5190                          | 3307            | 4133              | 26.89           | 2290            |
| LLM Runtime | 2012  | 32     | 48             | INT4      | INT8         | 32         | 29.09          | 5164                          | 8021            | 8923              | 29.18           | 5544            |
| LLM Runtime | 32    | 32     | 56             | INT4      | INT8         | 32         | 24.66          | 5243                          | 98.16           | 862.7             | 24.93           | 75.54           |
| LLM Runtime | 1024  | 32     | 56             | INT4      | INT8         | 32         | 27.07          | 5222                          | 3060            | 3899              | 27.38           | 2120            |
| LLM Runtime | 2012  | 32     | 56             | INT4      | INT8         | 32         | 29.56          | 5210                          | 7599            | 8515              | 29.85           | 5253            |
| GGML       | 32    | 32     | 32             | INT4      | INT8         | 32         | 33.69          | 3585                          | 393.24          | 1437              | 33.9            | 281.6           |
| GGML       | 1024  | 32     | 32             | INT4      | INT8         | 32         | 36.24          | 4389                          | 12702           | 13825             | 36.39           | 8775            |
| GGML       | 2012  | 32     | 32             | INT4      | INT8         | 32         | 39.19          | 5232                          | 27264           | 28479             | 39.44           | 18824           |
| GGML       | 32    | 32     | 48             | INT4      | INT8         | 32         | 30.34          | 5223                          | 291.84          | 1232              | 30.57           | 210             |
| GGML       | 1024  | 32     | 48             | INT4      | INT8         | 32         | 33.09          | 5137                          | 9206            | 10231             | 33.21           | 6362            |
| GGML       | 2012  | 32     | 48             | INT4      | INT8         | 32         | 37.34          | 5245                          | 21341           | 22499             | 37.66           | 14737           |
| GGML       | 32    | 32     | 56             | INT4      | INT8         | 32         | 31.62          | 5241                          | 262.3           | 1242              | 32              | 192.2           |
| GGML       | 1024  | 32     | 56             | INT4      | INT8         | 32         | 34.03          | 5193                          | 8363            | 9418              | 34.3            | 5781            |
| GGML       | 2012  | 32     | 56             | INT4      | INT8         | 32         | 36.94          | 5257                          | 18868           | 20013             | 37.66           | 13031           |


### Falcon-7B

| Backend    | Input | Output | Cores/Instance | Precision | Compute Type | Group Size | Next Token(ms) | Memory mean used (Top 50%) MB | First Token(ms) | Total Latency(ms) | P90 Latency(ms) | P99 Latency(ms) |
| ---------- | ----- | ------ | -------------- | --------- | ------------ | ---------- | -------------- | ----------------------------- | --------------- | ----------------- | --------------- | --------------- |
| LLM Runtime | 32    | 32     | 32             | INT4      | INT8         | 128        | 37.36          | 3797                          | 92.94           | 1251              | 37.69           | 75.88           |
| LLM Runtime | 1024  | 32     | 32             | INT4      | INT8         | 128        | 40.33          | 4707                          | 5507            | 6757              | 40.63           | 3813            |
| LLM Runtime | 32    | 32     | 48             | INT4      | INT8         | 128        | 35.84          | 4990                          | 88.29           | 1199              | 36.32           | 72.68           |
| LLM Runtime | 1024  | 32     | 48             | INT4      | INT8         | 128        | 37.95          | 4951                          | 5025            | 6201              | 38.14           | 3479            |
| LLM Runtime | 32    | 32     | 56             | INT4      | INT8         | 128        | 36.1           | 5019                          | 83.89           | 1202              | 36.36           | 69.19           |
| LLM Runtime | 1024  | 32     | 56             | INT4      | INT8         | 128        | 38.88          | 4993                          | 5432            | 6637              | 39.41           | 3761            |
| LLM Runtime | 32    | 32     | 32             | INT4      | INT8         | 32         | 39.15          | 4395                          | 146.7           | 1359              | 39.43           | 113.16          |
| LLM Runtime | 1024  | 32     | 32             | INT4      | INT8         | 32         | 41.61          | 5213                          | 6947            | 8237              | 42.54           | 4807            |
| LLM Runtime | 32    | 32     | 48             | INT4      | INT8         | 32         | 38.08          | 4980                          | 134.9           | 1315              | 38.23           | 105.1           |
| LLM Runtime | 1024  | 32     | 48             | INT4      | INT8         | 32         | 40.58          | 5085                          | 6847            | 8105              | 40.82           | 4737            |
| LLM Runtime | 32    | 32     | 56             | INT4      | INT8         | 32         | 38.33          | 5011                          | 142.4           | 1330              | 38.55           | 110.8           |
| LLM Runtime | 1024  | 32     | 56             | INT4      | INT8         | 32         | 40.87          | 5084                          | 6860            | 8127              | 41.18           | 4746            |
| GGML       | 32    | 32     | 32             | INT4      | INT8         | 32         | 38.44          | 4269                          | 458.3           | 1650              | 38.55           | 328.4           |
| GGML       | 1024  | 32     | 32             | INT4      | INT8         | 32         | 41.64          | 4997                          | 17585           | 18876             | 41.94           | 12147           |
| GGML       | 32    | 32     | 48             | INT4      | INT8         | 32         | 35.87          | 4971                          | 338.3           | 1450              | 36              | 244.7           |
| GGML       | 1024  | 32     | 48             | INT4      | INT8         | 32         | 38.68          | 5024                          | 13064           | 14263             | 39.06           | 9026            |
| GGML       | 32    | 32     | 56             | INT4      | INT8         | 32         | 36.22          | 5005                          | 318.9           | 1441              | 36.43           | 231.2           |
| GGML       | 1024  | 32     | 56             | INT4      | INT8         | 32         | 38.65          | 5045                          | 11943           | 13142             | 38.83           | 8253            |


### GPT-NEOX-20B

| Backend    | Input | Output | Cores/Instance | Precision | Compute Type | Group Size | Next Token(ms) | Memory mean used (Top 50%) MB | First Token(ms) | Total Latency(ms) | P90 Latency(ms) | P99 Latency(ms) |
| ---------- | ----- | ------ | -------------- | --------- | ------------ | ---------- | -------------- | ----------------------------- | --------------- | ----------------- | --------------- | --------------- |
| LLM Runtime | 32    | 32     | 32             | INT4      | INT8         | 128        | 68.77          | 10621                         | 234.18          | 2365              | 69.11           | 183.16          |
| LLM Runtime | 1024  | 32     | 32             | INT4      | INT8         | 128        | 76.55          | 12537                         | 9817            | 12190             | 77.06           | 6798            |
| LLM Runtime | 32    | 32     | 48             | INT4      | INT8         | 128        | 60.35          | 13639                         | 214.2           | 2085              | 60.59           | 167.34          |
| LLM Runtime | 1024  | 32     | 48             | INT4      | INT8         | 128        | 68.19          | 13524                         | 9213            | 11327             | 68.48           | 6378            |
| LLM Runtime | 32    | 32     | 56             | INT4      | INT8         | 128        | 80.16          | 13650                         | 221.5           | 2706              | 107.23          | 186.9           |
| LLM Runtime | 1024  | 32     | 56             | INT4      | INT8         | 128        | 88.48          | 13586                         | 10045           | 12788             | 111.93          | 6968            |
| LLM Runtime | 32    | 32     | 32             | INT4      | INT8         | 32         | 73.78          | 11970                         | 390.1           | 2308              | 74.13           | 308.2           |
| LLM Runtime | 1024  | 32     | 32             | INT4      | INT8         | 32         | 80.75          | 13871                         | 14993           | 17496             | 81.07           | 10370           |
| LLM Runtime | 32    | 32     | 48             | INT4      | INT8         | 32         | 68.17          | 13616                         | 348.6           | 2121              | 68.54           | 275.9           |
| LLM Runtime | 1024  | 32     | 48             | INT4      | INT8         | 32         | 74.84          | 13717                         | 15278           | 17598             | 75.24           | 10566           |
| LLM Runtime | 32    | 32     | 56             | INT4      | INT8         | 32         | 79.57          | 13638                         | 398.2           | 2467              | 103.79          | 324.2           |
| LLM Runtime | 1024  | 32     | 56             | INT4      | INT8         | 32         | 86.06          | 13703                         | 18119           | 20787             | 118.86          | 12541           |
| GGML       | 32    | 32     | 32             | INT4      | INT8         | 32         | 98.23          | 11660                         | 1403            | 4448              | 99.04           | 998.9           |
| GGML       | 1024  | 32     | 32             | INT4      | INT8         | 32         | 105.33         | 13686                         | 45434           | 48699             | 105.79          | 31382           |
| GGML       | 32    | 32     | 48             | INT4      | INT8         | 32         | 86.19          | 13582                         | 980             | 3651              | 86.74           | 703.1           |
| GGML       | 1024  | 32     | 48             | INT4      | INT8         | 32         | 93.79          | 13674                         | 32966           | 35873             | 94.4            | 22776           |
| GGML       | 32    | 32     | 56             | INT4      | INT8         | 32         | 92.36          | 13621                         | 1136            | 3999              | 119.08          | 823.6           |
| GGML       | 1024  | 32     | 56             | INT4      | INT8         | 32         | 95.49          | 13675                         | 39914           | 42874             | 115.03          | 27579           |


### Dolly-V2-3B

| Backend    | Input | Output | Cores/Instance | Precision | Compute Type | Group Size | Next Token(ms) | Memory mean used (Top 50%) MB | First Token(ms) | Total Latency(ms) | P90 Latency(ms) | P99 Latency(ms) |
| ---------- | ----- | ------ | -------------- | --------- | ------------ | ---------- | -------------- | ----------------------------- | --------------- | ----------------- | --------------- | --------------- |
| LLM Runtime | 32    | 32     | 32             | INT4      | INT8         | 128        | 21.84          | 2653                          | 78.37           | 755.43            | 22.29           | 61.18           |
| LLM Runtime | 1024  | 32     | 32             | INT4      | INT8         | 128        | 24.46          | 2653                          | 3725            | 4483              | 24.69           | 2578            |
| LLM Runtime | 32    | 32     | 48             | INT4      | INT8         | 128        | 22.76          | 2665                          | 81.26           | 786.95            | 23.06           | 63.31           |
| LLM Runtime | 1024  | 32     | 48             | INT4      | INT8         | 128        | 25.54          | 2677                          | 3399            | 4191              | 25.73           | 2354            |
| LLM Runtime | 32    | 32     | 56             | INT4      | INT8         | 128        | 22.02          | 2693                          | 78.17           | 760.6             | 22.14           | 61              |
| LLM Runtime | 1024  | 32     | 56             | INT4      | INT8         | 128        | 33.41          | 2693                          | 3799            | 4834              | 66.96           | 2643            |
| LLM Runtime | 32    | 32     | 32             | INT4      | INT8         | 32         | 22.5           | 2653                          | 95.91           | 793.2             | 22.78           | 73.27           |
| LLM Runtime | 1024  | 32     | 32             | INT4      | INT8         | 32         | 25.77          | 2653                          | 4374            | 5173              | 25.88           | 3026            |
| LLM Runtime | 32    | 32     | 48             | INT4      | INT8         | 32         | 23.77          | 2665                          | 97.84           | 834.5             | 23.84           | 75.06           |
| LLM Runtime | 1024  | 32     | 48             | INT4      | INT8         | 32         | 26.29          | 2728                          | 4361            | 5176              | 26.56           | 3018            |
| LLM Runtime | 32    | 32     | 56             | INT4      | INT8         | 32         | 23.79          | 2693                          | 88.55           | 826.7             | 23.91           | 68.58           |
| LLM Runtime | 1024  | 32     | 56             | INT4      | INT8         | 32         | 29.4           | 2725                          | 4822            | 5733              | 31.21           | 3348            |
| GGML       | 32    | 32     | 32             | INT4      | INT8         | 32         | 21.53          | 2653                          | 219.81          | 887.6             | 21.68           | 158.3           |
| GGML       | 1024  | 32     | 32             | INT4      | INT8         | 32         | 24.43          | 2653                          | 8011            | 8768              | 24.6            | 5535            |
| GGML       | 32    | 32     | 48             | INT4      | INT8         | 32         | 22.04          | 2665                          | 178.7           | 861.6             | 22.12           | 129.6           |
| GGML       | 1024  | 32     | 48             | INT4      | INT8         | 32         | 23.85          | 2693                          | 6342            | 7081              | 24.05           | 4384            |
| GGML       | 32    | 32     | 56             | INT4      | INT8         | 32         | 22.18          | 2693                          | 166.6           | 853.7             | 22.26           | 121.6           |
| GGML       | 1024  | 32     | 56             | INT4      | INT8         | 32         | 28.84          | 2703                          | 8715            | 9609              | 56.39           | 6034            |


### OPT-1.3B

| Backend    | Input | Output | Cores/Instance | Precision | Compute Type | Group Size | Next Token(ms) | Memory mean used (Top 50%) MB | First Token(ms) | Total Latency(ms) | P90 Latency(ms) | P99 Latency(ms) |
| ---------- | ----- | ------ | -------------- | --------- | ------------ | ---------- | -------------- | ----------------------------- | --------------- | ----------------- | --------------- | --------------- |
| LLM Runtime | 32    | 32     | 32             | INT4      | INT8         | 128        | 9.85           |  1680                         | 104.88          | 410.2             | 9.95            | 75.58           |
| LLM Runtime | 1024  | 32     | 32             | INT4      | INT8         | 128        | 11.38          | 1702                          | 3080            | 3433              | 11.83           | 2129            |
| LLM Runtime | 2012  | 32     | 32             | INT4      | INT8         | 128        | 13.15          | 2513                          | 7516            | 7924              | 13.41           | 5190            |
| LLM Runtime | 32    | 32     | 48             | INT4      | INT8         | 128        | 9.25           | 2709                          | 110.7           | 397.3             | 9.3             | 79.38           |
| LLM Runtime | 1024  | 32     | 48             | INT4      | INT8         | 128        | 11.1           | 2698                          | 3064            | 3408              | 11.15           | 2118            |
| LLM Runtime | 2012  | 32     | 48             | INT4      | INT8         | 128        | 12.77          | 2701                          | 8045            | 8441              | 13.02           | 5555            |
| LLM Runtime | 32    | 32     | 56             | INT4      | INT8         | 128        | 9.78           | 2742                          | 112.7           | 415.89            | 9.84            | 80.95           |
| LLM Runtime | 1024  | 32     | 56             | INT4      | INT8         | 128        | 16.96          | 2737                          | 3125            | 3650              | 54.16           | 2174            |
| LLM Runtime | 2012  | 32     | 56             | INT4      | INT8         | 32         | 16.69          | 2729                          | 7929            | 8447              | 24.51           | 5488            |
| LLM Runtime | 32    | 32     | 32             | INT4      | INT8         | 32         | 10.01          |  1703                         | 109.6           | 419.9             | 10.1            | 78.87           |
| LLM Runtime | 1024  | 32     | 32             | INT4      | INT8         | 32         | 11.71          | 1760                          | 3389            | 3752              | 11.8            | 2342            |
| LLM Runtime | 2012  | 32     | 32             | INT4      | INT8         | 32         | 13.58          | 2720                          | 8061            | 8482              | 13.63           | 5566            |
| LLM Runtime | 32    | 32     | 48             | INT4      | INT8         | 32         | 9.69           | 2709                          | 116.5           | 416.9             | 9.81            | 83.67           |
| LLM Runtime | 1024  | 32     | 48             | INT4      | INT8         | 32         | 11.51          | 2686                          | 3290            | 3647              | 11.55           | 2274            |
| LLM Runtime | 2012  | 32     | 48             | INT4      | INT8         | 32         | 13.09          | 2753                          | 8101            | 8507              | 13.14           | 5594            |
| LLM Runtime | 32    | 32     | 56             | INT4      | INT8         | 32         | 10.4           | 2742                          | 117.3           | 439.8             | 10.48           | 84.37           |
| LLM Runtime | 1024  | 32     | 56             | INT4      | INT8         | 32         | 15.65          | 2730                          | 3494            | 3979              | 37.89           | 2427            |
| LLM Runtime | 2012  | 32     | 56             | INT4      | INT8         | 32         | 20.52          | 2758                          | 8395            | 9031              | 55.67           | 5811            |
| GGML       | 32    | 32     | 32             | INT4      | INT8         | 32         | 8.47           |  1699                         | 170             | 432.6             | 8.88            | 120.12          |
| GGML       | 1024  | 32     | 32             | INT4      | INT8         | 32         | 10.07          | 1702                          | 4940            | 5252              | 10.13           | 3412            |
| GGML       | 2012  | 32     | 32             | INT4      | INT8         | 32         | 11.71          | 2709                          | 11741           | 12104             | 11.75           | 8105            |
| GGML       | 32    | 32     | 48             | INT4      | INT8         | 32         | 8.9            | 2709                          | 154.83          | 430.6             | 9.05            | 109.7           |
| GGML       | 1024  | 32     | 48             | INT4      | INT8         | 32         | 10.12          | 2669                          | 4409            | 4723              | 10.2            | 3046            |
| GGML       | 2012  | 32     | 48             | INT4      | INT8         | 32         | 12.16          | 2742                          | 11009           | 11386             | 12.19           | 7600            |
| GGML       | 32    | 32     | 56             | INT4      | INT8         | 32         | 9.48           | 2742                          | 152.31          | 446.04            | 9.56            | 108.14          |
| GGML       | 1024  | 32     | 56             | INT4      | INT8         | 32         | 14.39          | 2721                          | 5843            | 6289              | 27.99           | 4049            |
| GGML       | 2012  | 32     | 56             | INT4      | INT8         | 32         | 17.01          | 2751                          | 13001           | 13529             | 51.84           | 8989            |


### StarCoder-3B

| Backend    | Input | Output | Cores/Instance | Precision | Compute Type | Group Size | Next Token(ms) | Memory mean used (Top 50%) MB | First Token(ms) | Total Latency(ms) | P90 Latency(ms) | P99 Latency(ms) |
| ---------- | ----- | ------ | -------------- | --------- | ------------ | ---------- | -------------- | ----------------------------- | --------------- | ----------------- | --------------- | --------------- |
| LLM Runtime | 32    | 32     | 32             | INT4      | INT8         | 128        | 26.85          | 2868                          | 175.2           | 1007              | 27.12           | 129.3           |
| LLM Runtime | 32    | 32     | 48             | INT4      | INT8         | 128        | 26.78          | 2868                          | 172.1           | 1002              | 26.95           | 127.2           |
| LLM Runtime | 32    | 32     | 56             | INT4      | INT8         | 128        | 28.31          | 2763                          | 173.05          | 1050              | 28.53           | 128.7           |
| LLM Runtime | 32    | 32     | 32             | INT4      | INT8         | 32         | 27.8           | 2868                          | 200.74          | 1062              | 28.2            | 147.4           |
| LLM Runtime | 32    | 32     | 48             | INT4      | INT8         | 32         | 27.97          | 2896                          | 193.84          | 1060              | 28.12           | 142.9           |
| LLM Runtime | 32    | 32     | 56             | INT4      | INT8         | 32         | 29.16          | 2876                          | 195.67          | 1099              | 29.31           | 144.7           |
| GGML       | 32    | 32     | 32             | INT4      | INT8         | 32         | 26.57          | 2868                          | 368.5           | 1192              | 26.74           | 262.1           |
| GGML       | 32    | 32     | 48             | INT4      | INT8         | 32         | 26.5           | 2842                          | 310.5           | 1132              | 26.67           | 222.3           |
| GGML       | 32    | 32     | 56             | INT4      | INT8         | 32         | 27.17          | 2825                          | 293.92          | 1136              | 27.28           | 211.2           |


## LLM Finetuning

Environments:

PyTorch: 2.0.1+cpu


| Framework | Hidden Size | Dataset (Alpaca) | Concatenation | Nodes | PPN | Precision | LoRA | LoRA rank/alpha | Epoches | Time/Epoch | Total Time | TruthfulQA (mc1/mc2) | Global Batch Size | Learning Rate |
| --------- | ----------- | ----------------- | ------------- | ----- | --- | --------- | ---- | --------------- | ------- | ---------- | ---------- | -------------------- | ----------------- | ------------- |
| PyTorch  | 4096        | 13K               | Yes           | 1     | 1   | BF16      | Yes  | 8/16            | 3       | 3.2 Hour   | 9.6 Hours  | 0.30/0.45            | 128               | 1.00E-04      |
| PyTorch  | 4096        | 13K               | Yes           | 2     | 2   | BF16      | Yes  | 8/16            | 3       | 1.2 Hour   | 3.6 Hours  | 0.30/0.45            | 128               | 1.00E-04      |
| PyTorch  | 4096        | 13K               | Yes           | 4     | 2   | BF16      | Yes  | 8/16            | 3       | 0.67 Hour  | 2 Hours    | 0.30/0.45            | 128               | 1.00E-04      |


Intel Gaudi2 Environments: 

Driver version 1.13.0-ee32e42, synapse AI v1.13.0

We will release data soon.
