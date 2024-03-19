# Sparse GEMM VNNI

Sparse patterns must align with its target ISA, especially GEMM instructions.
VNNI introduces the following GEMM calculation:

> Description
>
> Multiply groups of four adjacent pairs of unsigned 8-bit integers in a with corresponding signed 8-bit integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the corresponding 32-bit integer in src, and store the packed 32-bit results in dst.
>
> Operation
>
> ```
> FOR j := 0 to 15
>     tmp1.word := Signed(ZeroExtend16(a.byte[4*j]) * SignExtend16(b.byte[4*j]))
>     tmp2.word := Signed(ZeroExtend16(a.byte[4*j+1]) * SignExtend16(b.byte[4*j+1]))
>     tmp3.word := Signed(ZeroExtend16(a.byte[4*j+2]) * SignExtend16(b.byte[4*j+2]))
>     tmp4.word := Signed(ZeroExtend16(a.byte[4*j+3]) * SignExtend16(b.byte[4*j+3]))
>     dst.dword[j] := src.dword[j] + tmp1 + tmp2 + tmp3 + tmp4
> ENDFOR
> dst[MAX:512] := 0
> ```

According to our kernel experiments, we defined the so-called **"4x1"** sparse pattern to fully utilize VNNI capability.
> Note:this kernel performs (transposed) weight multiplies (transposed) activation so that the "4x1" pattern here may be equivalent to the "1x4" pattern in other documents. The benefits of transposition are discussed in [the next section](#candidate-patterns).

As shown in the figure below, the sparse pattern in the weight (the left matrix) is 4x1, where 4 is in the output channel dimensions, 1 is in the input channel dimensions. In terms of a typical GEMM, we say that **4** is in **M** dimensions and **1** is in **K** dimension. For the rest part of this doc, we will use GEMM concept which usually uses **M**, **K**, **N** for the first, second and third dimension.

After compression, we concatenate each **4x1** block in the same row into some **4x4** blocks (there will be some padding in rows where the number of non-zero blocks is not a multiple of 4, as shown in the second and third row of blocks in the image). Then for each row (maybe including paddings) in a **4x4** block, we broadcast them to 4x16 elements, which meet the accumulation and parallelization dimensions of VNNI. For the activation (right) matrix, we pick 4 (or less if we added padding when preparing weight data) 1x16 blocks according to sparse indices and concatenate them into 4x16 blocks in column major. That is the second matrix we prepared for VNNI.

For a typical GEMM (MxKxN) problem, a high performance GEMM micro-kernel needs to tile in (mxkxn) which means taking m rows of matrix A and n columns of matrix B to improve the density of FMA instructions, therefore reducing bubbles in the assembly line. In the **4x4** block, the **"4"** is corresponding to the **m** in micro-kernels, and the **"4"** by the concatenate along rows is just for VNNI accumulation dimensions.

![image](../imgs/kernel_vnni_pattern_left_4x1.png)

## On the fly activation reordering

To use the VNNI instruction, the activation needs to be prepared in zmm registers in the order of 16x4. The following graph demonstrates this process:
![4x16_to_vnni_format](../imgs/4x16_to_vnni_format.png)
> Cell highlight colors indicate 4x 16-byte vectors.

We use [a talented solution from a smart guy](https://stackoverflow.com/a/64417691) to perform this transformation, which only uses four load and two swizzle instructions, adding minimal overhead to our kernel.

## Candidate patterns

Transposition is not a default option for matrix multiplication but a workaround we found after investigation. Without transposition, activation multiplies weight, and the major dimension of the left matrix (K) must be used for accumulation. Therefore, we need to use the N dim for parallelization which is sparse. Unfortunately, this leads to the difficulty of storing results in a non-consecutive memory location.

![image](../imgs/kernel_vnni_pattern_right_4x1.png)

Adopting the 1x16 pattern and using concatenation to get the accumulation dimension is also a bad idea, as it means loading 4 separate elements in a row on the dense matrix. Therefore, we must apply transposition so that the second dimension of the dense matrix is used for accumulation, leaving the leading dimension for parallelization.

![image](../imgs/kernel_vnni_pattern_right_1x16.png)

Given that transposition is necessary, an alternative pattern is **"1x4"**, where the concatenation cost is saved but missing tiling means lower density of micro-kernels, which will bring more harm for performance.

![image](../imgs/kernel_vnni_pattern_left_1x4.png)

> Comparison between **"1x4"** and **"4x1"**
>
> 1. **"4x1"** tiles along M dimensions, **"1x4"** cannot tile along this dimension.
> 2. **"4x1"** needs concatenation along K dimensions, **"1x4"** doesn't need concatenation. However, the concatenation happens during the compression, which is offline.

We performed a simulation by changing the **"4x1"** to **"2x1"**, which means tiling along the M dimensions becomes 2. As a result, the following table demonstrates about 1/3 perf drop, and we expect more performance drop with totally no tiling.

![image](../imgs/kernel_vnni_perf.png)

In conclusion, **"4x1"** brings higher performance which is about ~2(+) times against **"1x4"** (estimated from differences between tiling and no tiling.)
