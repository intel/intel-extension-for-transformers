import torch
torch.ops.load_library("build/libweight_only_jblasop.so")
raw_wei = torch.rand(512,512, dtype=torch.float)
activation = torch.rand(512,512, dtype=torch.float)
quant_wei = torch.ops.weight_only_jblasop.jblas_quantize(raw_wei,1,4,"sym",32,"fp32","vnni");
dst = torch.zeros(512,512,dtype=torch.float)
torch.ops.weight_only_jblasop.jblas_weights4block_f32_linear(activation,quant_wei,dst,512,512,512,512,512)
print(quant_wei)
print(activation)
print(dst)

