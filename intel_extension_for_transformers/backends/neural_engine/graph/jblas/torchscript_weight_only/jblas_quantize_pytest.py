import torch
torch.ops.load_library("build/libweight_only_jblasop.so")
A = torch.ones(262144, dtype=torch.float)
A=A.resize(512,512);
b = torch.ops.weight_only_jblasop.jblas_quantize(A,1,4,"sym",32,"fp32","vnni");
C = torch.rand(512,512,dtype=torch.float)
print(b)
print(C)

