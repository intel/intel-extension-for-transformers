# Static Compressed Buffer
- [Static Compressed Buffer](#static-compressed-buffer)
  - [Introduction](#introduction)
  - [How to Turn on `Static Compressed Buffer`](#how-to-turn-on-static-compressed-buffer)
  - [More Options](#more-options)

## Introduction
`Neural Engine` use `Cycle Buffer` as the default memory management tool in the inference session. It maintains a map which stores data pointer life count and size to reuse or malloc memory dynamically. However, the `GetMemory` function would become time-consuming in some ops (like slice). The `Cycle Buffer` also may lead memory waste when activation-tensor is very large. 
In this case, `Neural Engine` provides another choice of memory manager which called `Static Compressed Buffer`. It will allocate enough memory blocks during warmup process if users specify the maximum input shapes of the model. `Static Compressed Buffer` is faster, more stable and economical with very high compress-efficiency. 
We recommend you to turn on this memory management tool if you want to do some optimizations on you big models.

## How to Turn on `Static Compressed Buffer`
```python
# step 1. load an IR or convert a model from other frameworks
from intel_extension_for_transformers.backends.neural_engine.compile import compile
model = compile(model_path)
# setp 2. turn on the related option in graph
options = {'activation_mem_compression' : True}
graph.execution_options = options
# step 3. set max input shapes
# for example, bert-large (input_ids, token_type_ids, attention_mask)
graph.max_input_shapes_list = [ [[1, 128], [1, 128], [1, 128]] ]

# warm up
graph.inference(inputs)

# inference
for i in range(iters)
    graph.inference(inputs)
```

The most important part is setting the `graph.max_input_shapes_list`. This a model input shapes list which means it can receive one more input shapes if you don't know which shapes are the maximum. It's quite useful in some `Decoder-Only` models which have dynamic input shapes (for example, [LLAMA](https://huggingface.co/decapoda-research/llama-7b-hf)). For this case, users can pass several optional max input shapes.

For example:
```python
# LLAMA 7B
# 32 input tokens, 32 new output tokens
past_k_v_0 = [[1, 32, 1, 128] for i in range(64)]
past_k_v_1 = [[4, 32, 63, 128] for i in range(64)]
graph.max_input_shapes_list = [
                                # first iteration
                                [[1, 32]] + [[1, 33]] + past_k_v_0,
                                # the last iteration (beam size=4)
                                [[4, 1]] + [[4, 64]] + past_k_v_1,
                                ]
```

## More Options
For `Static Compressed Buffer`, `Neural Engine` supplies two extra options for debugging. They only works when turn on `activation_mem_compression`.
```python
options = {'activation_mem_compression' : True,
           # optional, save activation dag into the disk (path="activation_dag.yaml")
           'dump_activation_dag': True,
           # optional, no memory in-place in static compressed buffer
           'execution_mode': 'debug'
           }
```
