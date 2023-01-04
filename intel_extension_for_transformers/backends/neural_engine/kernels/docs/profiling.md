# Performance and Profiling

## Verbose
We support a brief verbose logger for kernel execution
```shell
SPARSE_LIB_VERBOSE=1 ./{executable}
sparselib_verbose,info,cpu,runtime:CPU,nthr:224                      # general info
sparselib_verbose,exec,cpu,sparse_matmul,shape_256_256_128,14.4658   # first kernel
sparselib_verbose,exec,cpu,sparse_matmul,shape_256_256_128,2.56982   # second kernel
```

## VTune
For advanced users we also support vtune profling for kernels execution through [ITT Tasks](https://www.intel.com/content/www/us/en/develop/documentation/vtune-help/top/api-support/instrumentation-and-tracing-technology-apis/basic-usage-and-configuration/viewing-itt-api-task-data.html), to enable it you can follow the instructions:

```shell
mkdir build
cd build
cmake .. -DNE_WITH_SPARSELIB_VTUNE=True
# if the path to vtune is not the default /opt/intel/oneapi/vtune/latest, you can determine the path manually like
# cmake .. -DNE_WITH_SPARSELIB_VTUNE=True -DCMAKE_VTUNE_HOME=/path/to/vtune
make -j
...
SPARSE_LIB_VTUNE=1 ./{executable}
```

We would recommend SSH methods in VTune to analyse details based on GUI.


## SDE
There is another way to verify the code generation itself via [SDE](https://www.intel.com/content/www/us/en/developer/articles/tool/software-development-emulator.html). SDE is also simulators for future intel CPU hardware so you can verify the implementation using feature ISAs **without** real hardware support:
```shell
SPARSE_LIB_DUMP=1 ./{executable}
path/to/sde/xed64 -64 -ir *.bin >> *.txt
```

<details>
  <summary>Click to see the whole assembly.txt!</summary>
  
```
XDIS 0: PUSH      BASE       53                       push rbx
XDIS 1: PUSH      BASE       55                       push rbp
XDIS 2: PUSH      BASE       4154                     push r12
XDIS 4: PUSH      BASE       4155                     push r13
XDIS 6: PUSH      BASE       4156                     push r14
XDIS 8: PUSH      BASE       4157                     push r15
XDIS a: DATAXFER  BASE       BD00040000               mov ebp, 0x400
XDIS f: DATAXFER  BASE       4C8B3F                   mov r15, qword ptr [rdi]
XDIS 12: DATAXFER  BASE       4C8B7708                 mov r14, qword ptr [rdi+0x8]
XDIS 16: DATAXFER  BASE       4C8B6F10                 mov r13, qword ptr [rdi+0x10]
XDIS 1a: LOGICAL   AVX512EVEX 62F17D48EFC0             vpxord zmm0, zmm0, zmm0
XDIS 20: LOGICAL   AVX512EVEX 62F15D48EFE4             vpxord zmm4, zmm4, zmm4
XDIS 26: LOGICAL   AVX512EVEX 62513D48EFC0             vpxord zmm8, zmm8, zmm8
XDIS 2c: LOGICAL   AVX512EVEX 62511D48EFE4             vpxord zmm12, zmm12, zmm12
XDIS 32: LOGICAL   AVX512EVEX 62F17548EFC9             vpxord zmm1, zmm1, zmm1
XDIS 38: LOGICAL   AVX512EVEX 62F15548EFED             vpxord zmm5, zmm5, zmm5
XDIS 3e: LOGICAL   AVX512EVEX 62513548EFC9             vpxord zmm9, zmm9, zmm9
XDIS 44: LOGICAL   AVX512EVEX 62511548EFED             vpxord zmm13, zmm13, zmm13
XDIS 4a: LOGICAL   AVX512EVEX 62F16D48EFD2             vpxord zmm2, zmm2, zmm2
XDIS 50: LOGICAL   AVX512EVEX 62F14D48EFF6             vpxord zmm6, zmm6, zmm6
XDIS 56: LOGICAL   AVX512EVEX 62512D48EFD2             vpxord zmm10, zmm10, zmm10
XDIS 5c: LOGICAL   AVX512EVEX 62510D48EFF6             vpxord zmm14, zmm14, zmm14
XDIS 62: DATAXFER  AVX512EVEX 62C17C481006             vmovups zmm16, zmmword ptr [r14]
XDIS 68: DATAXFER  AVX512EVEX 62C17C48104E01           vmovups zmm17, zmmword ptr [r14+0x40]
XDIS 6f: DATAXFER  AVX512EVEX 62C17C48105602           vmovups zmm18, zmmword ptr [r14+0x80]
XDIS 76: BROADCAST AVX512EVEX 62427D48183F             vbroadcastss zmm31, dword ptr [r15]
XDIS 7c: VFMA      AVX512EVEX 62927D40B8C7             vfmadd231ps zmm0, zmm16, zmm31
XDIS 82: VFMA      AVX512EVEX 62927540B8CF             vfmadd231ps zmm1, zmm17, zmm31
XDIS 88: VFMA      AVX512EVEX 62926D40B8D7             vfmadd231ps zmm2, zmm18, zmm31
XDIS 8e: BROADCAST AVX512EVEX 62427D48187F04           vbroadcastss zmm31, dword ptr [r15+0x10]
XDIS 95: VFMA      AVX512EVEX 62927D40B8E7             vfmadd231ps zmm4, zmm16, zmm31
XDIS 9b: VFMA      AVX512EVEX 62927540B8EF             vfmadd231ps zmm5, zmm17, zmm31
XDIS a1: VFMA      AVX512EVEX 62926D40B8F7             vfmadd231ps zmm6, zmm18, zmm31
XDIS a7: BROADCAST AVX512EVEX 62427D48187F08           vbroadcastss zmm31, dword ptr [r15+0x20]
XDIS ae: VFMA      AVX512EVEX 62127D40B8C7             vfmadd231ps zmm8, zmm16, zmm31
XDIS b4: VFMA      AVX512EVEX 62127540B8CF             vfmadd231ps zmm9, zmm17, zmm31
XDIS ba: VFMA      AVX512EVEX 62126D40B8D7             vfmadd231ps zmm10, zmm18, zmm31
XDIS c0: BROADCAST AVX512EVEX 62427D48187F0C           vbroadcastss zmm31, dword ptr [r15+0x30]
XDIS c7: VFMA      AVX512EVEX 62127D40B8E7             vfmadd231ps zmm12, zmm16, zmm31
XDIS cd: VFMA      AVX512EVEX 62127540B8EF             vfmadd231ps zmm13, zmm17, zmm31
XDIS d3: VFMA      AVX512EVEX 62126D40B8F7             vfmadd231ps zmm14, zmm18, zmm31
XDIS d9: DATAXFER  AVX512EVEX 62C17C48104603           vmovups zmm16, zmmword ptr [r14+0xc0]
XDIS e0: DATAXFER  AVX512EVEX 62C17C48104E04           vmovups zmm17, zmmword ptr [r14+0x100]
XDIS e7: DATAXFER  AVX512EVEX 62C17C48105605           vmovups zmm18, zmmword ptr [r14+0x140]
XDIS ee: BROADCAST AVX512EVEX 62427D48187F01           vbroadcastss zmm31, dword ptr [r15+0x4]
XDIS f5: VFMA      AVX512EVEX 62927D40B8C7             vfmadd231ps zmm0, zmm16, zmm31
XDIS fb: VFMA      AVX512EVEX 62927540B8CF             vfmadd231ps zmm1, zmm17, zmm31
XDIS 101: VFMA      AVX512EVEX 62926D40B8D7             vfmadd231ps zmm2, zmm18, zmm31
XDIS 107: BROADCAST AVX512EVEX 62427D48187F05           vbroadcastss zmm31, dword ptr [r15+0x14]
XDIS 10e: VFMA      AVX512EVEX 62927D40B8E7             vfmadd231ps zmm4, zmm16, zmm31
XDIS 114: VFMA      AVX512EVEX 62927540B8EF             vfmadd231ps zmm5, zmm17, zmm31
XDIS 11a: VFMA      AVX512EVEX 62926D40B8F7             vfmadd231ps zmm6, zmm18, zmm31
XDIS 120: BROADCAST AVX512EVEX 62427D48187F09           vbroadcastss zmm31, dword ptr [r15+0x24]
XDIS 127: VFMA      AVX512EVEX 62127D40B8C7             vfmadd231ps zmm8, zmm16, zmm31
XDIS 12d: VFMA      AVX512EVEX 62127540B8CF             vfmadd231ps zmm9, zmm17, zmm31
XDIS 133: VFMA      AVX512EVEX 62126D40B8D7             vfmadd231ps zmm10, zmm18, zmm31
XDIS 139: BROADCAST AVX512EVEX 62427D48187F0D           vbroadcastss zmm31, dword ptr [r15+0x34]
XDIS 140: VFMA      AVX512EVEX 62127D40B8E7             vfmadd231ps zmm12, zmm16, zmm31
XDIS 146: VFMA      AVX512EVEX 62127540B8EF             vfmadd231ps zmm13, zmm17, zmm31
XDIS 14c: VFMA      AVX512EVEX 62126D40B8F7             vfmadd231ps zmm14, zmm18, zmm31
XDIS 152: DATAXFER  AVX512EVEX 62C17C48104606           vmovups zmm16, zmmword ptr [r14+0x180]
XDIS 159: DATAXFER  AVX512EVEX 62C17C48104E07           vmovups zmm17, zmmword ptr [r14+0x1c0]
XDIS 160: DATAXFER  AVX512EVEX 62C17C48105608           vmovups zmm18, zmmword ptr [r14+0x200]
XDIS 167: BROADCAST AVX512EVEX 62427D48187F02           vbroadcastss zmm31, dword ptr [r15+0x8]
XDIS 16e: VFMA      AVX512EVEX 62927D40B8C7             vfmadd231ps zmm0, zmm16, zmm31
XDIS 174: VFMA      AVX512EVEX 62927540B8CF             vfmadd231ps zmm1, zmm17, zmm31
XDIS 17a: VFMA      AVX512EVEX 62926D40B8D7             vfmadd231ps zmm2, zmm18, zmm31
XDIS 180: BROADCAST AVX512EVEX 62427D48187F06           vbroadcastss zmm31, dword ptr [r15+0x18]
XDIS 187: VFMA      AVX512EVEX 62927D40B8E7             vfmadd231ps zmm4, zmm16, zmm31
XDIS 18d: VFMA      AVX512EVEX 62927540B8EF             vfmadd231ps zmm5, zmm17, zmm31
XDIS 193: VFMA      AVX512EVEX 62926D40B8F7             vfmadd231ps zmm6, zmm18, zmm31
XDIS 199: BROADCAST AVX512EVEX 62427D48187F0A           vbroadcastss zmm31, dword ptr [r15+0x28]
XDIS 1a0: VFMA      AVX512EVEX 62127D40B8C7             vfmadd231ps zmm8, zmm16, zmm31
XDIS 1a6: VFMA      AVX512EVEX 62127540B8CF             vfmadd231ps zmm9, zmm17, zmm31
XDIS 1ac: VFMA      AVX512EVEX 62126D40B8D7             vfmadd231ps zmm10, zmm18, zmm31
XDIS 1b2: BROADCAST AVX512EVEX 62427D48187F0E           vbroadcastss zmm31, dword ptr [r15+0x38]
XDIS 1b9: VFMA      AVX512EVEX 62127D40B8E7             vfmadd231ps zmm12, zmm16, zmm31
XDIS 1bf: VFMA      AVX512EVEX 62127540B8EF             vfmadd231ps zmm13, zmm17, zmm31
XDIS 1c5: VFMA      AVX512EVEX 62126D40B8F7             vfmadd231ps zmm14, zmm18, zmm31
XDIS 1cb: DATAXFER  AVX512EVEX 62C17C48104609           vmovups zmm16, zmmword ptr [r14+0x240]
XDIS 1d2: DATAXFER  AVX512EVEX 62C17C48104E0A           vmovups zmm17, zmmword ptr [r14+0x280]
XDIS 1d9: DATAXFER  AVX512EVEX 62C17C4810560B           vmovups zmm18, zmmword ptr [r14+0x2c0]
XDIS 1e0: BROADCAST AVX512EVEX 62427D48187F03           vbroadcastss zmm31, dword ptr [r15+0xc]
XDIS 1e7: VFMA      AVX512EVEX 62927D40B8C7             vfmadd231ps zmm0, zmm16, zmm31
XDIS 1ed: VFMA      AVX512EVEX 62927540B8CF             vfmadd231ps zmm1, zmm17, zmm31
XDIS 1f3: VFMA      AVX512EVEX 62926D40B8D7             vfmadd231ps zmm2, zmm18, zmm31
XDIS 1f9: BROADCAST AVX512EVEX 62427D48187F07           vbroadcastss zmm31, dword ptr [r15+0x1c]
XDIS 200: VFMA      AVX512EVEX 62927D40B8E7             vfmadd231ps zmm4, zmm16, zmm31
XDIS 206: VFMA      AVX512EVEX 62927540B8EF             vfmadd231ps zmm5, zmm17, zmm31
XDIS 20c: VFMA      AVX512EVEX 62926D40B8F7             vfmadd231ps zmm6, zmm18, zmm31
XDIS 212: BROADCAST AVX512EVEX 62427D48187F0B           vbroadcastss zmm31, dword ptr [r15+0x2c]
XDIS 219: VFMA      AVX512EVEX 62127D40B8C7             vfmadd231ps zmm8, zmm16, zmm31
XDIS 21f: VFMA      AVX512EVEX 62127540B8CF             vfmadd231ps zmm9, zmm17, zmm31
XDIS 225: VFMA      AVX512EVEX 62126D40B8D7             vfmadd231ps zmm10, zmm18, zmm31
XDIS 22b: BROADCAST AVX512EVEX 62427D48187F0F           vbroadcastss zmm31, dword ptr [r15+0x3c]
XDIS 232: VFMA      AVX512EVEX 62127D40B8E7             vfmadd231ps zmm12, zmm16, zmm31
XDIS 238: VFMA      AVX512EVEX 62127540B8EF             vfmadd231ps zmm13, zmm17, zmm31
XDIS 23e: VFMA      AVX512EVEX 62126D40B8F7             vfmadd231ps zmm14, zmm18, zmm31
XDIS 244: DATAXFER  AVX512EVEX 62D17C48114500           vmovups zmmword ptr [r13], zmm0
XDIS 24b: DATAXFER  AVX512EVEX 62D17C48116503           vmovups zmmword ptr [r13+0xc0], zmm4
XDIS 252: DATAXFER  AVX512EVEX 62517C48114506           vmovups zmmword ptr [r13+0x180], zmm8
XDIS 259: DATAXFER  AVX512EVEX 62517C48116509           vmovups zmmword ptr [r13+0x240], zmm12
XDIS 260: DATAXFER  AVX512EVEX 62D17C48114D01           vmovups zmmword ptr [r13+0x40], zmm1
XDIS 267: DATAXFER  AVX512EVEX 62D17C48116D04           vmovups zmmword ptr [r13+0x100], zmm5
XDIS 26e: DATAXFER  AVX512EVEX 62517C48114D07           vmovups zmmword ptr [r13+0x1c0], zmm9
XDIS 275: DATAXFER  AVX512EVEX 62517C48116D0A           vmovups zmmword ptr [r13+0x280], zmm13
XDIS 27c: DATAXFER  AVX512EVEX 62D17C48115502           vmovups zmmword ptr [r13+0x80], zmm2
XDIS 283: DATAXFER  AVX512EVEX 62D17C48117505           vmovups zmmword ptr [r13+0x140], zmm6
XDIS 28a: DATAXFER  AVX512EVEX 62517C48115508           vmovups zmmword ptr [r13+0x200], zmm10
XDIS 291: DATAXFER  AVX512EVEX 62517C4811750B           vmovups zmmword ptr [r13+0x2c0], zmm14
XDIS 298: POP       BASE       415F                     pop r15
XDIS 29a: POP       BASE       415E                     pop r14
XDIS 29c: POP       BASE       415D                     pop r13
XDIS 29e: POP       BASE       415C                     pop r12
XDIS 2a0: POP       BASE       5D                       pop rbp
XDIS 2a1: POP       BASE       5B                       pop rbx
XDIS 2a2: AVX       AVX        C5F877                   vzeroupper
XDIS 2a5: RET       BASE       C3                       ret
# end of text section.
# Errors: 0
#XED3 DECODE STATS
#Total DECODE cycles:        29220
#Total instructions DECODE: 68
#Total tail DECODE cycles:        236418
#Total tail instructions DECODE: 118
#Total cycles/instruction DECODE: 429.71
#Total tail cycles/instruction DECODE: 2003.54
```
  
 </details>

