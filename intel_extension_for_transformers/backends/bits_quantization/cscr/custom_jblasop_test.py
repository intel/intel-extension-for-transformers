import torch
torch.ops.load_library("build/libweight_only_jblasop.so")
activation = torch.rand(512,512, dtype=torch.float)
raw_wei = torch.rand(512,512, dtype=torch.float)
correct=torch.matmul(activation,raw_wei)
quant_wei = torch.ops.weight_only_jblasop.jblas_quantize(raw_wei,False,8,"sym",32,"fp32");
dst = torch.zeros(512,512,dtype=torch.float)
torch.ops.weight_only_jblasop.jblas_quantweight_f32_linear(activation,quant_wei,dst,512,512,512,512,512)
print(dst)
print(correct)