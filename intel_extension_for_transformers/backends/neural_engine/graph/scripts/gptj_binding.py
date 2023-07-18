#  Copyright (c) 2023 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from ctypes import *
import numpy as np

lib = cdll.LoadLibrary('./lib/libGptjPyBind.so')

init_gptj = lib.init_gptj
init_gptj.argtypes = [c_int, c_int, c_int, c_float, c_float, c_float, c_bool, c_int, c_char_p]
init_gptj.restype = c_void_p

gptj_in_all = init_gptj(1234, 32, 0, 1.0, 0.8, 1.5, False, 2048, b"../ne-q4_0.bin")

eval_gptj_char = lib.eval_gptj_char
eval_gptj_char.argtypes = [c_void_p, c_char_p, c_int, c_int, c_float, c_float, c_int]
eval_gptj_char.restype = c_char_p

#res = eval_gptj_char(gptj_in_all, b"she opened the door and saw", 32, 0, 1.0, 0.8, 1)

eval_gptj_ids = lib.eval_gptj_ids
eval_gptj_ids.argtypes = [c_void_p, np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'), c_int, c_int, c_int, c_float, c_float, c_int]
eval_gptj_ids.restype = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS')

#res = eval_gptj_ids(gptj_in_all, np.array([7091, 4721, 262, 3420, 290, 2497], dtype=np.int32), 6, 32, 0, 1.0, 0.8, 1)
res = eval_gptj_ids(gptj_in_all, np.array([7454, 2402, 257, 640, 11, 612, 11196, 257, 1310, 2576, 11, 508, 8288, 284, 423, 17545, 13, 1375, 2227, 284, 467, 284, 4113, 290, 1826, 649, 661, 11, 290, 423, 1257], dtype=np.int32), 31, 32, 0, 1.0, 0.8, 1)

ctypes_pntr = cast(res, POINTER(c_int))
res_np = np.ctypeslib.as_array(ctypes_pntr, shape=(31,))
exit_gptj = lib.exit_gptj
exit_gptj.argtypes = [c_void_p]
exit_gptj.restype = None

exit_gptj(gptj_in_all)
