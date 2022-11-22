# Profiling
## Introduction
In terms of improving the performance of the model  ,we should evaluate the performance of each operator(op)  during inference.
Intel Extension for Transformers supports  tracing the profiling of operator latency.
## Usage
### Example
run python
```shell
ENGINE_PROFILING=1  python run_executor.py --input_model=./model_and_tokenizer/int8-model.onnx  --mode=performance --batch_size=8 --seq_len=128
 ```
 or run C++
 ```shell
export ENGINE_PROFILING=1 
<intel_extension_for_transformers_folder>/intel_extension_for_transformers/backends/neural_engine/bin/neural_engine --batch_size=<batch_size> --iterations=<iterations> --w=<warmup> --seq_len=128 --config=./ir/conf.yaml --weight=./ir/model.bin
 ```

 ## Result
 ### We will get a profiling form,such as follows. This form is divided into three sections.

 #### Part 1
- Some arguments for sparse include weight shape,sparse ratio and target performance ratio.
- Users can set the parameter pref ratio independently to calculate sparse op performance.
 
| Arguments  | Weight shape   |  90% 4x1 perf ratio  |80% 4x1 perf ratio|70% 4x1 Perf ratio|
| --------   | :-----:  | :----:  | :----:  | :----:  |
|   value    |  256x256    |**4(optional)**|**2.5(optional)**|**2(optional)**|
|    value   |  256x1024    |**4.5(optional)**|**3(optional)**|**2.5(optional)**|
|    value   |  1024x256  |**5(optional)**|**3.5(optional)**|**3(optional)**|
|description       |  Shape of weight for "matmul" or "innerproduct"  |The op's sparse ratio is 90%, and the performance ratio is "dense op latency"/ "sparse op latency" , representing the performance improvement of the op after sparse. This parameter can be set by the user.|Same as 90% 4x1 perf ratio |Same as 90% 4x1 perf ratio|
#### Part 2
- All operator's profiling, such as operator type ,input tensor ,output tensor and latency .Let's take "innerproduct" as an example.
- In this form, we can auto calculate the sparse op performance by customized sparse ratio.

|   Argument    |  Value  |  Additional description  |
| :--------:  | :-----:  | :----:  |
|  operator type     |  InnerProduct  |  None   |
|    post op   |   gelu_tanh |   In order to improve the performance of inference, we use multiple ops as one op for inference|
|   operator name    |   Add_37  |  None   |
|    input tensor name   |  116:0;641:0;bert.encoder.layer.0.attention.self.key.bias:0  |  The name of input tensor(include multi inputs)   |
|    input shape   |  1024x256;256x256;256  |  The shape of input tensor(include multi inputs)   |
|   input dtype    |    fp32;fp32;fp32|   None  |
|   output tensor name    |   Add_37:0|   None  |
|    output shape   |     1024x256 |   The shape of output tensor  |
|   output dtype    |   fp32   |   None  |
|   weight shape    |   256x256   |  Shape of weight for "matmul" or "innerproduct" |
|    weight sparse ratio  |  0.00%    |    The current sparse ratio for weight  |
|    sparse support   |    TRUE  |   Whether to support sparse  |
|   operator latency (ms)    |   0.075  |  The latency before sparse   |
|   **aim to weight sparse ratio**    |  **70%(optional)**  |  **Target weight sparse ratio ,option: 90%,80%,70%,etc**  |
|   pref ratio  |   2  |  Auto look up part 1 form |
|   aim to sparse latency(ms)  |   0.0375 |  Target sparse latency = "operator latency(0.075)"/"perf ratio(2)"(auto calculate)|

#### Part 3
- Performance comparison  of dense and sparse  networks.

|Arguments|Value|Description|
 |-----------|:--------:|:--------:|
 |total latency(ms)|4.512|The latency for the entire network to inference once before  sparse|
 |total aim to sparse latency(ms)|2.185|The latency for the entire network to inference once after sparse|
 |sparse support latency(ms)|3.127|The latency for all  operators that support sparse to inference once before sparse|
 |aim to sparse support latency(ms)|0.801| The latency for all  operators that support sparse to inference once after sparse|
 |sparse support latency ratio|0.693|The ratio of the latency of the operator  before sparse to the latency for the entire network to inference once|
 |aim to sparse support latency ratio|0.366|The ratio of latency of the operator after sparse to the latency for the entire network to inference once|
 
## Cautions 
 - We have obtained a form in csv format, and we can modify the form content of part1 to obtain the desired performance, but after modification, we need to save the form format as "xlsx".
