##  Copyright (c) 2023 Intel Corporation
##
##  Licensed under the Apache License, Version 2.0 (the "License");
##  you may not use this file except in compliance with the License.
##  You may obtain a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
##  Unless required by applicable law or agreed to in writing, software
##  distributed under the License is distributed on an "AS IS" BASIS,
##  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##  See the License for the specific language governing permissions and
##  limitations under the License.

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
