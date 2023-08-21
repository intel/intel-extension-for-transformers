import torch
import inspect
from functools import wraps
torch.ops.load_library("build/libweight_only_jblasop.so")

def capture_args(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(f)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        arg_strs = []
        for name, value in bound_args.arguments.items():
            arg_strs.append(f'{name}={value}')
        result = ', '.join(arg_strs)
        print(result)
        return f(*args, **kwargs)
    return wrapper

@capture_args
def test(m,n,k,blocksize,compute_type,quant_dtype,transpose,dump_tensor_info=False):
    activation = torch.rand(m,k, dtype=torch.float)
    wei_row=k
    wei_col=n
    if transpose:
        wei_row,wei_col=wei_col,wei_row;     
    raw_wei=torch.rand(wei_row,wei_col,dtype=torch.float)
    bias=torch.rand(n,dtype=torch.float)
    bias*=10
    quant_wei=torch.ops.weight_only_jblasop.jblas_quantize(raw_wei,transpose,"sym",blocksize,compute_type,quant_dtype);
    torch.ops.weight_only_jblasop.jblas_symqdq_weight(raw_wei,transpose,quant_dtype,blocksize)
    if transpose:
        raw_wei=torch.transpose(raw_wei,0,1)
    trans_correct=torch.matmul(activation,raw_wei)
    trans_dst = torch.zeros(m,n,dtype=torch.float)
    if dump_tensor_info:
        print("==========bias========")
        print(bias)
    torch.ops.weight_only_jblasop.jblas_quantweight_f32_linear_with_bias(activation,quant_wei,bias,trans_dst,m,n,k,k,n,compute_type,quant_dtype)
    if dump_tensor_info:
        print("==============transformat with bias result===============")
        print(trans_dst)
        print("~~~~~~~~~~~~~~~~~~")
        print(trans_correct+bias)
    ok=True
    ok =ok&torch.allclose(trans_dst,trans_correct+bias,rtol=0.03)
    torch.ops.weight_only_jblasop.jblas_quantweight_f32_linear_without_bias(activation,quant_wei,trans_dst,m,n,k,k,n,compute_type,quant_dtype)
    if(dump_tensor_info):
        print("==============transformat without bias result===============")
        print(trans_dst)
        print("~~~~~~~~~~~~~~~~~~")
        print(trans_correct)
    ok=ok&torch.allclose(trans_dst,trans_correct,rtol=0.03)
    if ok:
        print("ok.")
    else:
        print("fail")

test(2,3,32,32,"fp32","s8",True)
test(2,3,32,32,"fp32","s4_clip",True)
test(2,3,32,32,"int8","s4_clip",True)
test(2,3,32,32,"fp32","s4_fullrange",True)
test(2,3,32,32,"fp32","s8",False)
test(2,3,32,32,"fp32","s4_clip",False)
test(2,3,32,32,"int8","s4_clip",False)
test(2,3,32,32,"fp32","s4_fullrange",False)
test(2,3,32,32,"fp32","nf4",True)
test(2,3,32,32,"fp32","nf4",False)