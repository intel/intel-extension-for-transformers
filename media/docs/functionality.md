[README](/README.md#documentation) > **Functionality**

# Functionality

- int8 - 8 bits integer
- fp16 - half-precision floating-point
- fp32 - single-precision floating-point
- bf16 - bfloat16 (Brain Floating Point Format)
- bf32 - bfloat32
- tf32 - tensor float32
- N - Row Major matrix
- T - Column Major Matrix
- {N,T} x {N,T} - All combinations, i.e., NN, NT, TN, TT
- FPU - Floating Point Processing Unit
- XMX - Matrix Engine

## GEMM

The following tables summarize GEMM kernel's feature set, organized API, data type and layout. Hyperlinks to relevant examples/unit tests demonstrate how specific template instances may be defined.

|**API** | **Data Type**                  | **Compute Engine** |**Layouts**            | **Unit Test**    |
|-----------------|--------------------------------|------------------------|------------------|------------------|
| **GEMM**        |  `int8 * int8 => { int8, int32 }`      |XMX| {N,T} x {N,T} => N |  [example](/tests/integration/gemm/int8) |
| **GEMM**        |  `bf16 * bf16 => { bf16, fp32 }`       |XMX| {N,T} x {N,T} => N |  [example](/tests/integration/gemm/bf16) |
| **GEMM**        |  `fp16 * fp16 => { fp16, fp32 }`       |XMX| {N,T} x {N,T} => N |  [example](/tests/integration/gemm/fp16) |
| **GEMM**        |  `tf32 * tf32 => { tf32, fp32 }`       |XMX| {N,T} x {N,T} => N |  [example](/tests/integration/gemm/tf32) |
| **GEMM**        |  `fp32 * fp32 =>  fp32 `               |FPU| {N,T} x {N,T} => N |  [example](/tests/integration/gemm/fp32) |

## Epilogue

The following table summarizes epilogue APIs, organized by API and data type.Hyperlinks to relevant examples/unit tests demonstrate how specific template instances may be defined.

|**API** | **Data Type**                   | **Unit Test**    |
|-----------------|--------------------------------|------------------------|
| **Bias Add**        |  `{ int8, bf16, bf32, fp16, fp32, tf32 }`       |  [example](/examples/03_gemm_relu_bias) |
| **GELU Forward**        |  `{ int8, bf16, bf32, fp16, fp32, tf32 }`       |  [example](/tests/unit/epilogue_tile_op) |
| **GELU Backward**        |  `{ int8, bf16, bf32, fp16, fp32, tf32 }`       |  [example](/tests/unit/epilogue_tile_op) |
| **RELU**        |  `{ int8, bf16, bf32, fp16, fp32, tf32 }`       |  [example](/examples/03_gemm_relu_bias) |
| **Residual Add**        |  `{ int8, bf16, bf32, fp16, fp32, tf32 }`       |  [example](/tests/unit/epilogue_tile_op) |

## Copyright
Copyright (c) 2022-2023 Intel Corporation Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
