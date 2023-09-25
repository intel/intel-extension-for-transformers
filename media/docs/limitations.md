[README](/README.md#Limitations) > **Limitations**

# Limitations

Some XeTLA APIs have limitations due to hardware restrictions or software design.
XeTLA added checkings for these restrictions and end users could get error messages
when they touched the limitations. We added the checkings in kernel, group, subgroup levels.

## Limitations And Checkers

<table>
 <tr>
  <td><b><p style="text-align: center;">
Level</p></b></td>
  <td><b><p style="text-align: center;">Feature</p></b></td>
  <td><b><p style="text-align: center;">Category</p></b></td>
  <td><b><p style="text-align: center;">Restriction</p></b></td>
  <td><b><p style="text-align: center;">API</p></b></td>
 </tr>
 <tr height=43 style='height:32.5pt'>
  <td rowspan=2 class=xl158 width=64 style='border-top:none;width:48pt'>kernel</td>
  <td rowspan=2 >gemm</td>
  <td class=xl152 width=118 style='border-top:none;border-left:none;width:89pt'>general
  1d</td>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt'><i>refer
  to <b>table 1-1</b></i><span style='mso-spacerun:yes'> </span></td>
  <td class=xl152 width=440 style='border-top:none;border-left:none;width:330pt'>template
  <typename T><br>
    class general_1d<gpu_arch::Xe, T>::check_alignment(T *base, uint32_t pitch)</td>
 </tr>
 <tr height=43 style='height:32.5pt'>
  <td class=xl152 width=118 style='border-top:none;border-left:none;width:89pt'>block
  2d</td>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt'><i>refer
  to <b>table 1-2</b></i><span style='mso-spacerun:yes'> </span></td>
  <td class=xl152 width=440 style='border-top:none;border-left:none;width:330pt'>template
  <typename T><br>
    class block_2d<gpu_arch::Xe, T>::check_tensor(
            uint64_t base, uint32_t width, uint32_t height, uint32_t pitch)</td>
 </tr>
 <tr height=86 style='height:64.5pt'>
  <td rowspan=18 class=xl158 width=64 style='border-top:none;width:48pt'>group</td>
  <td rowspan=6 class=xl152 width=74 style='border-top:none;width:56pt;
  box-sizing: border-box;border:var(--borderColor-default, var(--color-border-default))'>FPU</td>
  <td class=xl152 width=118 style='border-top:none;border-left:none;width:89pt;
  box-sizing: border-box;border:var(--borderColor-default, var(--color-border-default))'>data
  type</td>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt;
  box-sizing: border-box;border:var(--borderColor-default, var(--color-border-default))'><i>dtype_mma_a,
  dtype_mma_b, dtype_mma_acc must be float type</i></td>
  <td class=xl152 width=440 style='border-top:none;border-left:none;width:330pt'>template
  <typename dtype_a, typename dtype_b, typename dtype_mma_a,<br>
    <span style='mso-spacerun:yes'>                </span>typename dtype_mma_b,
  typename dtype_mma_acc><br>
    <span style='mso-spacerun:yes'>        </span>struct check_dtype_default</td>
 </tr>
 <tr height=65 style='height:48.5pt'>
  <td class=xl152 width=118 style='border-top:none;border-left:none;width:89pt'>memory</td>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt;
  box-sizing: border-box;border:var(--borderColor-default, var(--color-border-default))'><i>Don't
  support matrixA and matrixB load from local memory</td>
  <td class=xl152 width=440 style='border-top:none;border-left:none;width:330pt;
  box-sizing: border-box;border:var(--borderColor-default, var(--color-border-default))'>template
  <mem_layout mem_layout_a, mem_layout mem_layout_b,<br>
    <span style='mso-spacerun:yes'>                </span>mem_space
  mem_space_a, mem_space mem_space_b><br>
    <span style='mso-spacerun:yes'>        </span>struct check_memory_default</td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td rowspan=4 class=xl152 width=118 style='border-top:none;width:89pt'>tile
  size</td>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt;
  box-sizing: border-box;border:var(--borderColor-default, var(--color-border-default))'><i>(block_size_x_b
  % (64 / sizeof(dtype_mma))) == 0</td>
  <td rowspan=4 class=xl152 width=440 style='border-top:none;width:330pt;
  box-sizing: border-box;border:var(--borderColor-default, var(--color-border-default))'>template
  <typename arch_attr, typename dtype_mma, int tile_size_x_a,<br>
    <span style='mso-spacerun:yes'>                </span>int tile_size_y_a,
  int block_size_x_a, int block_size_y_a,<br>
    <span style='mso-spacerun:yes'>                </span>int tile_size_x_b,
  int tile_size_y_b, int block_size_x_b,<br>
    <span style='mso-spacerun:yes'>                </span>int
  block_size_y_b><br>
    <span style='mso-spacerun:yes'>        </span>struct
  check_tile_size_default</td>
 </tr>
 <tr height=43 style='height:32.5pt'>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt'><i>(tile_size_x_a
  % block_size_x_a) == 0</td>
 </tr>
 <tr height=43 style='height:32.5pt'>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt'><i>(tile_size_y_b
  % block_size_y_b) == 0</td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt'><i>block_size_x_a
  == block_size_y_b</td>
 </tr>
 <tr height=43 style='height:32.5pt'>
  <td rowspan=12 class=xl152 width=74 style='border-top:none;width:56pt'>XMX</td>
  <td rowspan=2 class=xl152 width=118 style='border-top:none;width:89pt;
  box-sizing: border-box;border:var(--borderColor-default, var(--color-border-default))'>data
  type</td>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt;
  box-sizing: border-box;border:var(--borderColor-default, var(--color-border-default))'><i>dtype_mma_a
  should be the same as dtype_mma_b in xe arch</td>
  <td rowspan=2 class=xl152 width=440 style='border-top:none;width:330pt;
  box-sizing: border-box;border:var(--borderColor-default, var(--color-border-default))'>template
  <typename dtype_a, typename dtype_b, typename dtype_mma_a,<br>
    <span style='mso-spacerun:yes'>                </span>typename
  dtype_mma_b><br>
    <span style='mso-spacerun:yes'>        </span>struct check_dtype_default</td>
 </tr>
 <tr height=43 style='height:32.5pt'>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt'><i>not
  support fp32<->fp8, since it will meet a lot of HW limitations</td>
 </tr>
 <tr height=43 style='height:32.5pt'>
  <td rowspan=2 class=xl152 width=118 style='border-top:none;width:89pt'>memory</td>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt;
  box-sizing: border-box;border:var(--borderColor-default, var(--color-border-default))'><i>matA
  load from local memory, then matA should be row-major</td>
  <td rowspan=2 class=xl152 width=440 style='border-top:none;width:330pt;
  box-sizing: border-box;border:var(--borderColor-default, var(--color-border-default))'>template
  <mem_layout mem_layout_a, mem_layout mem_layout_b,<br>
    <span style='mso-spacerun:yes'>                </span>mem_space
  mem_space_a, mem_space mem_space_b><br>
    <span style='mso-spacerun:yes'>        </span>struct check_memory_default</td>
 </tr>
 <tr height=43 style='height:32.5pt'>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt'><i>matB
  load from local memory, then matB should be row-major</td>
 </tr>
 <tr height=43 style='height:32.5pt'>
  <td rowspan=8 class=xl152 width=118 style='border-top:none;width:89pt'>tile
  size</td>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt;
  box-sizing: border-box;border:var(--borderColor-default, var(--color-border-default))'><i>tile_size_x_a
  should be a multiple of mma_k</td>
  <td rowspan=8 class=xl152 width=440 style='border-top:none;width:330pt;
  box-sizing: border-box;border:var(--borderColor-default, var(--color-border-default))'>template
  <typename arch_attr, typename dtype_mma, int tile_size_x_a,<br>
    <span style='mso-spacerun:yes'>                </span>int tile_size_y_a,
  int block_size_x_a, int block_size_y_a,<br>
    <span style='mso-spacerun:yes'>                </span>int tile_size_x_b,
  int tile_size_y_b, int block_size_x_b,<br>
    <span style='mso-spacerun:yes'>                </span>int
  block_size_y_b><br>
    <span style='mso-spacerun:yes'>        </span>struct
  check_tile_size_default</td>
 </tr>
 <tr height=43 style='height:32.5pt'>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt'><i>block_size_x_a
  should be equal to mma_k</td>
 </tr>
 <tr height=43 style='height:32.5pt'>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt'><i>tile_size_y_a
  should be a multiple of mma_m</td>
 </tr>
 <tr height=43 style='height:32.5pt'>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt'><i>block_size_y_a
  should be a multiple of mma_m</td>
 </tr>
 <tr height=43 style='height:32.5pt'>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt'><i>tile_size_x_b
  should be a multiple of mma_n</td>
 </tr>
 <tr height=43 style='height:32.5pt'>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt'><i>block_size_x_b
  should be equal to mma_n</td>
 </tr>
 <tr height=43 style='height:32.5pt'>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt'><i>tile_size_y_b
  should be a multiple of mma_k</td>
 </tr>
 <tr height=43 style='height:32.5pt'>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt'><i>block_size_y_b
  should be a multiple of mma_k</td>
 </tr>
 <tr height=43 style='height:32.5pt'>
  <td rowspan=20 class=xl158 width=64 style='border-top:none;width:48pt'>subgroup</td>
  <td rowspan=9 class=xl152 width=74 style='border-top:none;width:56pt;
  box-sizing: border-box;border:var(--borderColor-default, var(--color-border-default))'>load</td>
  <td rowspan=2 class=xl152 width=118 style='border-top:none;width:89pt;
  box-sizing: border-box;border:var(--borderColor-default, var(--color-border-default))'>global
  2d</td>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt;
  box-sizing: border-box;border:var(--borderColor-default, var(--color-border-default))'><i>For
  VNNI transform, the maximum block width is 16 width</td>
  <td rowspan=2 class=xl152 width=440 style='border-top:none;width:330pt;
  box-sizing: border-box;border:var(--borderColor-default, var(--color-border-default))'>template
  <typename dtype, typename mem_dtype><br>
    struct check_load<gpu_arch::Xe, dtype, mem_dtype> {<br>
    <span style='mso-spacerun:yes'>    </span>template <bool mem_transform,
  size_t block_size_x><br>
    <span style='mso-spacerun:yes'>    </span>struct global_2d</td>
 </tr>
 <tr height=43 style='height:32.5pt'>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt'><i>max_block_width
  should be a multiply of block size x</td>
 </tr>
 <tr height=86 style='height:64.5pt'>
  <td class=xl152 width=118 style='border-top:none;border-left:none;width:89pt'>global
  1d</td>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt;
  box-sizing: border-box;border:var(--borderColor-default, var(--color-border-default))'><i>sizeof(mem_dtype) == 4 || sizeof(mem_dtype) == 8</td>
  <td class=xl152 width=440 style='border-top:none;border-left:none;width:330pt;
  box-sizing: border-box;border:var(--borderColor-default, var(--color-border-default))'>template
  <typename dtype, typename mem_dtype><br>
    struct check_load<gpu_arch::Xe, dtype, mem_dtype> {<br>
    <span style='mso-spacerun:yes'>    </span>
    <span style='mso-spacerun:yes'>    </span>struct global_1d</td>
 </tr>
 <tr height=65 style='height:48.5pt'>
  <td rowspan=5 class=xl152 width=118 style='border-top:none;width:89pt'>local
  scatter</td>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt;
  box-sizing: border-box;border:var(--borderColor-default, var(--color-border-default))'><i>only
  support row major in local load, you can use local store to do the transpose</td>
  <td rowspan=5 class=xl152 width=440 style='border-top:none;width:330pt;
  box-sizing: border-box;border:var(--borderColor-default, var(--color-border-default))'>template
  <typename dtype, typename mem_dtype><br>
    struct check_load<gpu_arch::Xe, dtype, mem_dtype> {<br>
    <span style='mso-spacerun:yes'>    </span>template <mem_layout memory_layout, size_t block_size_x, size_t tile_bytes,
            size_t min_bytes, size_t block_bytes, size_t num_channel_x,
            size_t num_channel><br>
    <span style='mso-spacerun:yes'>    </span>struct local_scatter</td>
 </tr>
 <tr height=43 style='height:32.5pt'>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt'><i>load
  size should at least DW aligned</td>
 </tr>
 <tr height=43 style='height:32.5pt'>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt'><i>bytes
  per row should be a multiply of sizeof load_dtype</td>
 </tr>
 <tr height=43 style='height:32.5pt'>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt'><i>(tile_bytes
  % min_bytes) == 0 && (block_bytes % min_bytes) == 0</td>
 </tr>
 <tr height=65 style='height:48.5pt'>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt'><i>The
  number of simd channel x should be greater than 0 and less than num_channel</td>
 </tr>
 <tr height=86 style='height:64.5pt'>
  <td class=xl152 width=118 style='border-top:none;border-left:none;width:89pt'>local
  1d</td>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt;
  box-sizing: border-box;border:var(--borderColor-default, var(--color-border-default))'><i>tile
  1d only support D32/D64</td>
  <td class=xl152 width=440 style='border-top:none;border-left:none;width:330pt;
  box-sizing: border-box;border:var(--borderColor-default, var(--color-border-default))'>template
  <typename dtype, typename mem_dtype><br>
    struct check_load<gpu_arch::Xe, dtype, mem_dtype> {<br>
    <span style='mso-spacerun:yes'>    </span>
    <span style='mso-spacerun:yes'>    </span>struct local_1d</td>
 </tr>
 <tr height=86 style='height:64.5pt'>
  <td rowspan=11 class=xl152 width=74 style='border-top:none;width:56pt'>store</td>
  <td class=xl152 width=118 style='border-top:none;border-left:none;width:89pt;
  box-sizing: border-box;border:var(--borderColor-default, var(--color-border-default))'>global
  2d</td>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt;
  box-sizing: border-box;border:var(--borderColor-default, var(--color-border-default))'><i>max_block_width
  should be a multiply of block size x</td>
  <td class=xl152 width=440 style='border-top:none;border-left:none;width:330pt;
  box-sizing: border-box;border:var(--borderColor-default, var(--color-border-default))'>template
  <typename dtype, typename mem_dtype><br>
    struct check_store<gpu_arch::Xe, dtype, mem_dtype> {<br>
    <span style='mso-spacerun:yes'>    </span>template <size_t
  block_size_x><br>
    <span style='mso-spacerun:yes'>    </span>struct global_2d</td>
 </tr>
 <tr height=86 style='height:64.5pt'>
  <td class=xl152 width=118 style='border-top:none;border-left:none;width:89pt'>global
  1d</td>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt;
  box-sizing: border-box;border:var(--borderColor-default, var(--color-border-default))'><i>tile
  1d only support D32/D64</td>
  <td class=xl152 width=440 style='border-top:none;border-left:none;width:330pt;
  box-sizing: border-box;border:var(--borderColor-default, var(--color-border-default))'>template
  <typename dtype, typename mem_dtype><br>
    struct check_store<gpu_arch::Xe, dtype, mem_dtype> {<br>
    <span style='mso-spacerun:yes'>    </span>
    <span style='mso-spacerun:yes'>    </span>struct global_1d</td>
 </tr>
 <tr height=65 style='height:48.5pt'>
  <td rowspan=4 class=xl152 width=118 style='border-top:none;width:89pt'>global
  atomic</td>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt;
  box-sizing: border-box;border:var(--borderColor-default, var(--color-border-default))'><i>for
  global atomic add, we only support fp32,fp64,uin32_t,uint64_t,int</td>
  <td rowspan=4 class=xl152 width=440 style='border-top:none;width:330pt;
  box-sizing: border-box;border:var(--borderColor-default, var(--color-border-default))'>template
  <typename dtype, typename mem_dtype><br>
    struct check_store<gpu_arch::Xe, dtype, mem_dtype> {<br>
    <span style='mso-spacerun:yes'>    </span>template <size_t tile_bytes, size_t min_store_bytes, size_t block_bytes,
            size_t num_channel_x, size_t num_channel><br>
    <span style='mso-spacerun:yes'>    </span>struct global_atomic</td>
 </tr>
 <tr height=65 style='height:48.5pt'>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt'><i>(tile_bytes
  % min_store_bytes) == 0 && (block_bytes % min_store_bytes) == 0</td>
 </tr>
 <tr height=65 style='height:48.5pt'>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt'><i>The
  number of simd channel x should be greater than 0 and less than num_channel</td>
 </tr>
 <tr height=43 style='height:32.5pt'>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt'><i>Only
  support DW and QW atomic add</td>
 </tr>
 <tr height=43 style='height:32.5pt'>
  <td rowspan=2 class=xl152 width=118 style='border-top:none;width:89pt'>local
  scatter</td>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt;
  box-sizing: border-box;border:var(--borderColor-default, var(--color-border-default))'><i>(tile_bytes
  % min_bytes) == 0 && (block_bytes % min_bytes) == 0</td>
  <td rowspan=2 class=xl152 width=440 style='border-top:none;width:330pt;
  box-sizing: border-box;border:var(--borderColor-default, var(--color-border-default))'>template
  <typename dtype, typename mem_dtype><br>
    struct check_store<gpu_arch::Xe, dtype, mem_dtype> {<br>
    <span style='mso-spacerun:yes'>    </span>template <size_t tile_bytes, size_t min_bytes, size_t block_bytes,
            size_t num_channel_x, size_t num_channel><br>
    <span style='mso-spacerun:yes'>    </span>struct local_scatter</td>
 </tr>
 <tr height=65 style='height:48.5pt'>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt'><i>The
  number of simd channel x should be greater than 0 and less than num_channel</td>
 </tr>
 <tr height=65 style='height:48.5pt'>
  <td rowspan=2 class=xl152 width=118 style='border-top:none;width:89pt'>local
  scatter vnni col</td>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt;
  box-sizing: border-box;border:var(--borderColor-default, var(--color-border-default))'><i>(tile_bytes
  % min_store_bytes) == 0 && (block_bytes % min_store_bytes) == 0</td>
  <td rowspan=2 class=xl152 width=440 style='border-top:none;width:330pt;
  box-sizing: border-box;border:var(--borderColor-default, var(--color-border-default))'>template
  <typename dtype, typename mem_dtype><br>
    struct check_store<gpu_arch::Xe, dtype, mem_dtype> {<br>
    <span style='mso-spacerun:yes'>    </span>template <size_t tile_bytes, size_t min_store_bytes, size_t block_bytes,
            size_t num_channel_x, size_t num_channel><br>
    <span style='mso-spacerun:yes'>    </span>struct local_scatter_vnni_col</td>
 </tr>
 <tr height=65 style='height:48.5pt'>
  <td class=xl152 width=214 style='border-top:none;border-left:none;width:161pt'><i>The
  number of simd channel x should be greater than 0 and less than num_channel</td>
 </tr>
</table>

## Table 1-1

<table>
 <tr>
  <td><b><p style="text-align: center;">Addr Type</b></td>
  <td><b><p style="text-align: center;">Data Size</b></td>
  <td><b><p style="text-align: center;">Address Size</b></td>
  <td><b><p style="text-align: center;">Addr Alignment</b></td>
  <td><b><p style="text-align: center;">Vector Size</b></td>
  <td><b><p style="text-align: center;">Transpose</b></td>
  <td><b><p style="text-align: center;">SIMT Mask</b></td>
 </tr>
 <tr>
  <td>global</td>
  <td>D8U32, D16U32, D32, D64</td>
  <td>A32, A64</td>
  <td>byte</td>
  <td>1</td>
  <td>off</td>
  <td>1, 2, 4, 8, 16, 32</td>
 </tr>
 <tr>
  <td>global</td>
  <td>D32,
  D64</td>
  <td>A32, A64</td>
  <td>data size</td>
  <td>2, 3, 4, 8</td>
  <td>off</td>
  <td>1, 2, 4, 8, 16, 32</td>
 </tr>
 <tr>
  <td>global</td>
  <td>D32, D64</td>
  <td>A32, A64</td>
  <td>data size</td>
  <td>1, 2, 3, 4, 8, 16, 32, 64</td>
  <td>on</td>
  <td>1</td>
 </tr>
 <tr>
  <td>slm</td>
  <td>D8U32, D16U32, D32, D64</td>
  <td>A16, A32</td>
  <td>byte</td>
  <td>1</td>
  <td>off</td>
  <td>1, 2, 4, 8, 16, 32</td>
 </tr>
 <tr>
  <td>slm</td>
  <td>D32, D64</td>
  <td>A16, A32</td>
  <td>data size</td>
  <td>2, 3, 4, 8</td>
  <td>off</td>
  <td>1, 2, 4, 8, 16, 32</td>
 </tr>
 <tr>
  <td>slm</td>
  <td>D32, D64</td>
  <td>A16, A32</td>
  <td>data size</td>
  <td>1, 2, 3, 4, 8, 16, 32, 64</td>
  <td>on</td>
  <td>1</td>
 </tr>
</table>

## Table 1-2

<table>
 <tr>
  <td><b><p style="text-align: center;">Category</p></b></td>
  <td><b><p style="text-align: center;">Data Size</p></b></td>
  <td><b><p style="text-align: center;">Restrictions</p></b></td>
 </tr>
 <tr>
  <td>base address</td>
  <td>U64</td>
  <td>base address must be dword aligned.</td>
 </tr>
 <tr>
  <td>surface width</td>
  <td>U32</td>
  <td>1. only 24 bits are supported for surface width field, bits [31:24] are ignored
  by the hardware.<br>
    2. surface width (encoded_value + 1) must be equal or greater than 64B.</td>
 </tr>
 <tr>
  <td>surface height</td>
  <td>U32</td>
  <td>only 24 bits are supported for surface height field, bits [31:24] are ignored by
  the hardware.</td>
 </tr>
 <tr>
  <td>surface pitch</td>
  <td>U32</td>
  <td>1. pitch must be greater or equal to width.<br>
    2. only 24 bits are supported for surface pitch field, bits [31:24] are
  ignored by the hardware.<br>
    3. surface pitch (encoded_value + 1) must be equal or greater than
  64B.<br>
    4. surface pitch (encoded_value + 1) must be a multiple of OW (16 bytes).</td>
 </tr>
 <tr>
  <td>block start x</td>
  <td>S31</td>
  <td>for data-size d8, block start x must be a multiple of 4. for data-size of d16,
  block start x must be a multiple of 2.</td>
 </tr>
 <tr>
  <td>block start y</td>
  <td>S31</td>
  <td>N/A</td>
 </tr>
 <tr>
  <td>block width</td>
  <td>U8</td>
  <td>1.this field value must be between 0-63.<br>
    2. block width (encoded_value + 1)
  mulitiplied by the element size (bytes) must be a multiple of 4 bytes. which
  means, for element size of 1 byte, the block width (encoded_value+1) should
  be a multiple of 4, and for element size of 2 bytes, the block width should
  be a multiple of 2.</td>
 </tr>
 <tr>
  <td>block height</td>
  <td>U8</td>
  <td>this field value must be between 0-31.</span></td>
 </tr>
 <tr>
  <td>array length</td>
  <td>U4</td>
  <td>1. the range of this field must be in 0-3.<br>
    2. this field must be zero for 2d block store messages.</td>
 </tr>
</table>

# Copyright

Copyright (c) 2022-2023 Intel Corporation Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
