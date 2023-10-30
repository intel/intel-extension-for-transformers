from setuptools import setup
import torch
import intel_extension_for_pytorch
from torch.xpu.cpp_extension import DPCPPExtension, DpcppBuildExtension

setup(
    name='gbits_quantize',
    ext_modules=[
        DPCPPExtension('gbits_quantize', [
            'gbits_quantize.cpp',
        ])
    ],
    cmdclass={
        'build_ext': DpcppBuildExtension
    })

setup(
    name='gbits_dequantize',
    ext_modules=[
        DPCPPExtension(
        name='gbits_dequantize', 
        extra_compile_args={'cxx': ['-std=c++17', '-fPIC']},
        sources=['gbits_dequantize.cpp'],
        )
    ],
    cmdclass={
        'build_ext': DpcppBuildExtension
    })


setup(
    name='gbits_linear',
    ext_modules=[
        DPCPPExtension(
                name='gbits_linear',
                sources=['gbits_linear.cpp'],
                extra_compile_args={'cxx': ['-std=c++17', '-fPIC']},
                library_dirs=["./build/"],
                libraries=["xetla_linear"],
                ),
    ],
    cmdclass={
        'build_ext': DpcppBuildExtension
    },
    )
