from setuptools import setup
import torch
import intel_extension_for_pytorch
from torch.xpu.cpp_extension import DPCPPExtension, DpcppBuildExtension

# setup(
#     name='gbits_quantize',
#     ext_modules=[
#         DPCPPExtension('gbits_quantize', [
#             'gbits_quantize.cpp',
#         ])
#     ],
#     cmdclass={
#         'build_ext': DpcppBuildExtension
#     })
# setup(
#     name='gbits_dequantize',
#     ext_modules=[
#         DPCPPExtension('gbits_dequantize', [
#             'gbits_dequantize.cpp',
#         ])
#     ],
#     cmdclass={
#         'build_ext': DpcppBuildExtension
#     })

# setup(
#     name='gbits_linear',
#     ext_modules=[
#         DPCPPExtension(
#                 name='gbits_linear',
#                 sources=['gbits_linear.cpp'],
#                 extra_compile_args={'cxx': ['-g', '-std=c++20', '-fPIC']})

#     ],
#     cmdclass={
#         'build_ext': DpcppBuildExtension
#     },
#     include_dirs=['../../../library/xetla',
#                   '../../../library/xetla/include']
#     )

# setup(
#     name='xetla_linear',
#     ext_modules=[
#         DPCPPExtension(
#                 name='xetla_linear',
#                 sources=['xetla_linear.cpp'],
#                 extra_compile_args={'cxx': ['-std=c++20', '-fPIC' , "-fsycl"]})


#     ],
#     cmdclass={
#         'build_ext': DpcppBuildExtension
#     },
#     include_dirs=['../../../library/xetla',
#                   '../../../library/xetla/include']
#     )

setup(
    name='gbits_linear',
    ext_modules=[
        DPCPPExtension(
                name='gbits_linear',
                sources=['gbits_linear_new.cpp'],
                extra_compile_args={'cxx': ['-std=c++17', '-fPIC']},
                library_dirs=["/home/sunjiwei/code/intel-extension-for-transformers/intel_extension_for_transformers/llm/operator/cscr/xpu/build/"],
                libraries=["xetla_linear"],
                ),
    ],
    cmdclass={
        'build_ext': DpcppBuildExtension
    },
    )
