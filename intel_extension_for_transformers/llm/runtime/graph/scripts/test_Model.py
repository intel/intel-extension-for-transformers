import numpy as np
import torch
from intel_extension_for_transformers.llm.runtime.graph import Model
import intel_extension_for_transformers.llm.runtime.graph.chatglm_cpp as cpp_model
# model = Model()

# int_weight = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.int8)
int_weight = torch.load("int_weight.pth").detach().numpy()
# scale = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
scale = torch.load("scale.pth").detach().numpy()
# import pdb; pdb.set_trace()
dst = np.zeros((4096, 4096), dtype=np.int8)

# model.init_from_bin("mpt", "/mnt/disk2/data/zhenweil/codes/ggml/mpt_ne.bin", max_new_tokens=20, num_beams=1, do_sample=True, top_k=40, top_p=0.95)
cpp_model.Model.np_jblas_qpack(np.left_shift(int_weight, 4), scale, dst)

# 打印C++函数返回的指针值
print(int_weight)
print(dst)
print(np.right_shift(dst, 4))
import pdb; pdb.set_trace()

import struct
# num = struct.pack('b', -128)
# 打开一个文件以二进制写入
# with open('output.bin', 'wb') as f:
#     for i in range(len(dst)):
#         f.write(struct.pack('b', dst[i]))


# float32 convert to jblas tensor
weight_f32 = torch.randn(4096, 4096).numpy()
dst_f32 = np.zeros((4096, 4096 * 2), dtype=np.int8)
ssize = cpp_model.Model.np_jblas_quantize(weight_f32, dst_f32)
# import pdb; pdb.set_trace()
print(ssize)
print(weight_f32)
print(dst_f32)