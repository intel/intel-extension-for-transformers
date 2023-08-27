# Profiling

1. [Introduction](#Introduction)  
2. [Profling API](#Profling-API)  
3. [Profiling Examples](#Profiling-Examples)  
3.1 [Parts of CSV Profiling](#Parts-of-CSV-Profiling)  
&ensp;&ensp;3.1.1 [Sparse Ratio Setting Part](#Sparse-Ratio-Setting-Part)  
&ensp;&ensp;3.1.2 [Operator Profiling Part](#Operator-Profiling-Part)  
&ensp;&ensp;3.1.3 [Total Profiling Part](#Total-Profiling-Part)  
3.2 [Levels of JSON Profiling](#Levels-of-JSON-Profiling)  
&ensp;&ensp;3.2.1 [Model Level](#Model-Level)  
&ensp;&ensp;3.2.2 [Iteration Level](#Iteration-Level)  
&ensp;&ensp;3.2.3 [Operator Level](#Operator-Level)  

## Introduction
In order to better analyze the performance of the model, we could evaluate the performance of each operator in inference. There is a profling tool in Neural Engine to collect the latency of operators.

## Profling API

### You can get profile only with ENGINE_PROFILING=1 before running model by python/c++ API.

Let's take [bert_mini_sst2](../../../../examples/huggingface/pytorch/text-classification/deployment/sst2/bert_mini) for example. You can follow the steps in example README.md and just add ENGINE_PROFILING=1 before run executor like this:
run python
```shell
ENGINE_PROFILING=1 python run_executor.py --input_model=./model_and_tokenizer/int8-model.onnx --mode=performance --batch_size=8 --seq_len=128
 ```
 or run C++
 ```shell
ENGINE_PROFILING=1 neural_engine --batch_size=<batch_size> --iterations=10 --w=5 --seq_len=128 --config=./ir/conf.yaml --weight=./ir/model.bin
 ```
 Of course, you can also **export ENGINE_PROFILING=1** before running. After that, there will be a folder named engine_profliling including profiling_csv and profiling_trace under your current path. The profiling_csv records average latancy of each operator and the whole model. You can also set perf ratio to estimate perfromance improvement automatically. The profiling_trace records the latency of each iteration and more operate details. You can just load it on **chrome://tracing/** and view. If you want to analyze more of performance, you can deal with it as a json format file on your way.
 >**Note**: In multiple instances case, you need to tell how many instances will run, just **export INST_NUM=<inst num>**. And as for multiple instances, we will get profiling_<time>_<inst_count>.csv/.json of each instance.

## Profiling Examples

### Parts of CSV Profiling

#### Sparse Ratio Setting Part
- Some arguments for sparse include weight shape, sparse ratio and target performance ratio.
- Users can set the parameter pref ratio to estimate sparse operator performance improvement semi-automatically.
 
| Arguments  | Weight shape   |  90% 4x1 perf ratio  |80% 4x1 perf ratio|70% 4x1 Perf ratio|
| --------   | :-----:  | :----:  | :----:  | :----:  |
|   value    |  256x256   |**4(settable)**|**2.5(settable)**|**2(settable)**|
|   value    |  256x1024  |**4.5(settable)**|**3(settable)**|**2.5(settable)**|
|   value    |  1024x256  |**5(settable)**|**3.5(settable)**|**3(settable)**|
|description       |  Shape of weight for "matmul" or "innerproduct"  |The op's sparse ratio is 90%, and the performance ratio is "dense op latency"/ "sparse op latency" , representing the performance improvement of the op after sparse. This parameter can be set by the user.|Same as 90% 4x1 perf ratio |Same as 90% 4x1 perf ratio|
#### Operator Profiling Part
- All operator's profiling, such as operator type, input tensor, output tensor and latency. Let's take "innerproduct" as an example.
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
|   **aim to weight sparse ratio**    |  **70%(settable)**  |  **Target weight sparse ratio, option: 90%,80%,70%,etc**  |
|   pref ratio  |   2  |  Auto look up part 1 form |
|   aim to sparse latency(ms)  |   0.0375 |  Target sparse latency = "operator latency(0.075)"/"perf ratio(2)"(auto calculate)|

#### Total Profiling Part
- Performance comparison of dense and sparse networks.

|Arguments|Value|Description|
 |-----------|:--------:|:--------:|
 |total latency(ms)|4.512|The latency for the entire network to inference once before sparse|
 |total aim to sparse latency(ms)|2.185|The latency for the entire network to inference once after sparse|
 |sparse support latency(ms)|3.127|The latency for all operators that support sparse to inference once before sparse|
 |aim to sparse support latency(ms)|0.801| The latency for all operators that support sparse to inference once after sparse|
 |sparse support latency ratio|0.693|The ratio of the latency of the operator before sparse to the latency for the entire network to inference once|
 |aim to sparse support latency ratio|0.366|The ratio of latency of the operator after sparse to the latency for the entire network to inference once|
 
>**Note**: We have obtained a form in csv format, and we can modify the form content of part1 to obtain the desired performance, but after modification, we need to save the form format as "xlsx".

### Levels of JSON Profiling

#### Model Level
The model_inference level records the latency of model from start to end.

```shell
Title    model_inference
Category    inference
User Friendly Category    other
Start    0.000 ms
Wall Duration    17.138 ms
```

#### Iteration Level
The iteration level records the latency of each iteration from start to end.
```shell
Title    Iteration4
Category    iteration
User Friendly Category    other
Start    0.000 ms
Wall Duration    8.726 ms
```

#### Operator Level
The operator level records the latency of per operator in per iteration from start to end. And there are also details in Args. The reshape_time means the latency happened to prepare tensor shape phrase. It will be 0ms in iteration more than 0 static shape case. The forward_time means latency of kernel calculation. And you can also see input/output tensor_name, tensor_type, tensor_shape. As for attributes, it will include some parameters in this operator such as padding in conv, post op such as sum and permutation information such src1_perm:1,0 meaning that input tensor 1 transposes 0 and 1 axis.
```shell
Title    Add_284
Category    InnerProduct
User Friendly Category    other
Start    12.028 ms
Wall Duration    0.044 ms
Args
reshape_time    "0.004ms"
forward_time    "0.044ms"
input_tensor_name    "onnx::MatMul_357:0,onnx::MatMul_358:0,
                    bert.encoder.layer.1.output.dense.bias:0,
                    onnx::MatMul_346:0"
input_type    "fp32,fp32,f32,"
input_shape    "64*1024,256*1024,256"
output_tensor_name    "input.44:0"
output_type    "fp32"
output_shape    "64*256"
attributes    "append_op:sum;src1_perm:1,0"
```
