# Export to ONNX

1. [Introduction](#introduction)

2. [Supported Model Export Matrix](#supported-model-export-matrix)

3. [Examples](#examples)

    3.1. [Export to FP32 ONNX Model](#export-to-fp32-onnx-model)

    3.2. [Export to BF16 ONNX Model](#export-to-bf16-onnx-model)

    3.3. [Export to INT8 ONNX Model](#export-to-int8-onnx-model)


## Introduction
We support exporting PyTorch models into ONNX models with our well-desighed API `trainer.export_to_onnx`. Users can get FP32 (Float precision 32 bit), BF16 (Bfloat 16 bit) and INT8 (Integer 8 bit) ONNX model with the same interface.


## Supported Model Export Matrix

| Input Model | Export FP32 | Export BF16 | Export INT8 |
| --- | --- | --- | --- |
| FP32 PyTorch Model | &#10004; | &#10004; | / |
| INT8 PyTorch Model <br> (PostTrainingDynamic) | / | / | &#10004; |
| INT8 PyTorch Model <br> (PostTrainingStatic) | / | / | &#10004; |
| INT8 PyTorch Model <br> (QuantizationAwareTraining) | / | / | &#10004; |


## Examples

### Export to FP32 ONNX Model

If `export_to_onnx` is called before quantization, we will fetch the FP32 model and export it into a ONNX model.

```py
trainer.export_to_onnx(
    save_path=None, 
    [opset_version=14,]
    [do_constant_folding=True,]
    [verbose=True,]
)
```

### Export to BF16 ONNX Model

If the flag: `enable_bf16` is True, you will get an ONNX model with BFloat16 weights for ['MatMul', 'Gemm'] node type. This FP32 + BF16 ONNX model can be accelerated by our [executor](../intel_extension_for_transformers/backends/neural_engine/) backend.

### API usage

```py
trainer.enable_bf16 = True
trainer.export_to_onnx(
    save_path=None, 
    [opset_version=14,]
    [do_constant_folding=True,]
    [verbose=True,]
)
```

### Export to INT8 ONNX Model

If `export_to_onnx` is called after quantization, we will fetch the FP32 PyTorch model, convert it into ONNX model and do onnxruntime quantization based on pytorch quantization configuration.

```py
trainer.export_to_onnx(
    save_path=None,
    [quant_format='QDQ'/'Qlinear',]
    [dtype='S8S8'/'U8S8'/'U8U8',]
    [opset_version=14,]
)
```

### **For executor backend**
Our executor backend provides highly optimized performance for INT8 `MatMul` node type and `U8S8` datatype. Therefore, we suggest users to enable the flag `enable_executor` before export int8 ONNX model for executor backend.

```py
trainer.enable_executor = True
```
