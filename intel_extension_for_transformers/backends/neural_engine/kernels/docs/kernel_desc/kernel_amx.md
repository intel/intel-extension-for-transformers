# Design of AMX Kernel

As we all know, AMX ISA introduces the `tdpbf16ps`, which does **16x32** matrix times **32x16** matrix as the following:
```
FOR m := 0 TO dst.rows - 1
	tmp := dst.row[m]
	FOR k := 0 TO (a.colsb / 4) - 1                                                 // colsb => bytes per col, in BF16 case k = [0, 16)
		FOR n := 0 TO (dst.colsb / 4) - 1                                           // colsb => bytes per col, in BF16 case n = [0, 16)
			tmp.fp32[n] += FP32(a.row[m].bf16[2*k+0]) * FP32(b.row[k].bf16[2*n+0])
			tmp.fp32[n] += FP32(a.row[m].bf16[2*k+1]) * FP32(b.row[k].bf16[2*n+1])
		ENDFOR
	ENDFOR
	write_row_and_zero(dst, m, tmp, dst.colsb)
ENDFOR
zero_upper_rows(dst, dst.rows)
zero_tileconfig_start()
```
Like the **VNNI**, **AMX-BF16** needs re-layout the right matrix as following:


![image](../imgs/kernel_amx_bf16x16_relayout.png)


As a result, as a successor to **AVX512** series sparse pattern, **AMX** pattern split the matrix into many **1x16** blocks and then concatenate them to meet AMX 32x16 requirements.


![image](../imgs/kernel_amx_bf16x16_calc.png)

Let the left matrix in the above image be A and the right be B. In our case, A is transposed weight(sparse), B is transposed activation. A can be compressed offline or before the inference, so we could directly use `tileloadd` instruction for 32 nonzero blocks in A (they’re stored consecutively in memory). For B, We need 32 rows of 16 consecutive values for one `tdpbf16ps`. This may be good because 16 values in one row are consecutive and can be loaded via `vmovdqu32` (through we are handling BF16). However, there are two tradeoff. The first is that the 32x16 tiles of B need to be reordered to AMX layout as the images shows. The second is that the activation matrix needs to be transposed, which may be more time-consuming

Alternatively, considering A as activation and B as weight can save a lot of time because weight is fixed and can be compressed, concatenated and reordered offline. However, the loading of activation is a disaster because all 32 **16x1** blocks are not consecutive can will greatly impact performance, related experiments WIP.

Then the key to the question is that how to re-layout activation on the fly. Luckily, a really smart guy gave [a brief and talented solution](https://stackoverflow.com/questions/64409634/4-way-bytewise-interleave-4x-16-byte-vectors-from-memory-with-avx512). Although the answer is for VNNI layout, but also applicable for our questions. The related code is as the following:

```cpp
const static __m512i mask = _mm512_set_epi16(31,15,30,14,29,13,28,12,27,11,26,10,25,9,24,8,23,7,22,6,21,5,20,4,19,3,18,2,17,1,16,0);
__m256i lo = _mm256_loadu_epi(...);
__m256i li = _mm256_loadu_epi(...);
__m512i vec = _mm512_inserti32x8(_mm512_castsi256_si512(lo), li, 1);
__m512i permuted = _mm512_permutexvar_epi16(mask, vec);
_mm512_storeu_epi32(..., permuted);
```
