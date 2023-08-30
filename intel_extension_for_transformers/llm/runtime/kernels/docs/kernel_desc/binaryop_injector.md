# Binary Injectors

-- [Introduction](#introduction)
-- [Framework Features](#framework-features)
-  - [param\_type.hpp](#param_typehpp)
-  - [operator\_desc.hpp](#operator_deschpp)
-  - [jit\_binaryop\_injector.hpp](#jit_binaryop_injectorhpp)
-- [Usage](#usage)
-  - [Developer's Perspective](#developers-perspective)
-  - [User's Perspective](#users-perspective)

## Introduction
Some DL operators need two changeable source operands `src1` & `src2`, apply a series of computations(e.g. add, horizontal max or combine some ops to a new op like dynamic quantization), and store the result to the dst. We call these operators **binaryop**.<br />In general, the data in `src1` and `src2` can be stored in registers or memory. Binaryop also can apply op-fusion optimization. Take the embedding layer in the bert model as an example, the gather op can be fused with the next op(binaryadd) and reduce the overhead of moving data from memory to the SIMD register in one of the source operands.<br />
For implementing the binaryop-fusion, we design a new injector, `binaryop_injector`. The source operands `src1` & `src2` are from SIMD register and memory, We will make the binaryop_injector as a component of the postop_injector so users can apply the eltwiseops/binaryops they configured in postop-chain sequentially in the future.<br />
For kernel developers, the using step of `binaryop_injector` is quite similar to `eltwise_injector`, but more easily. `binaryop_injector` does not need to call the `escape` function because our current binaryops are quite simple and don't need too many SIMD registers to do some computation. kernel developer also doesn't need to prepare LUT because no const value will be used in binaryop now. Please notice that except `compute_vector`, binaryop_injector also exposes some **"simple-arithmetic-op"** to the kernel developer. The purpose is to reduce the time of browsing ISA doc, developers only need to tell the injector op_type and data_type then binaryop_injector can insert appropriate instruction automatically(e.g. dt=s8, op_type=add insert `vpaddb`; dt=fp32, op_type=add insert `vaddps`).<br />
For Transformers-accelerated Libraries's user(engine developer), if they want to config binaryop-fusion, they need to config the `binaryop_attr` filed in `operator_desc`. Unlike `eltwise_injector`, the user needs to call the `set_binaryop_list` function to set the `binaryop_attr`, rather than setting it in the construct function. The purpose is to reduce API changing and improve scalability. If more fields need to be added in the future, I suggest using the get/set function set them.
<a name="nC8IX"></a>
## Framework Features
<a name="gGo6C"></a>
### param_type.hpp
Some new classes/structs will be introduced. The most important class is binaryop_attr, which indicates what type of algorithm we want to apply and the ptrs we needed.`static_addr` is used for normal binaryop, `scale` and `zp` are used for per-channel quantization/dequantization.  
Please notice that Transformers-accelerated Libraries's users must free the ptrs like static_addr on their own, Transformers-accelerated Libraries will not free these ptrs.
```cpp
enum class binaryop_alg : uint8_t { undef, add, sub, mul, per_channel_quant, per_channel_dequant };

class binaryop_attr {
 public:
  void* static_addr;
  float* scale;
  float* zp;
  binaryop_alg op_alg;
  data_type op_dt;

  binaryop_attr(binaryop_alg alg, data_type dt) : op_alg(alg), op_dt(dt) {
    static_addr = nullptr;
    scale = nullptr;
    zp = nullptr;
  }
  binaryop_attr(void* ptr, binaryop_alg alg, data_type dt) : static_addr(ptr), op_alg(alg), op_dt(dt) {}
  void set_scale(float* scale) { this->scale = scale; }
  void set_zp(float* zp) { this->zp = zp; }
};
```
<a name="JzRdj"></a>
### operator_desc.hpp
The member `binaryop_list_` store the binaryop_attr which user wants to apply.<br />Add two new methods to get/set `binaryop_list_`.
```cpp
public:
  void set_binaryop_list(const std::vector<binaryop_attr>& binaryop_list) { this->binaryop_list_ = binaryop_list; }
  inline const std::vector<binaryop_attr>& get_binaryop_list() const { return binaryop_list_; };
private:
  std::vector<binaryop_attr> binaryop_list_;
```
<a name="aUhwk"></a>
### jit_binaryop_injector.hpp
The APIs `binaryop_injector` exposes to users are as follows:<br />`binary_injector_init` used to initial the `binaryop_injector`

`set_mask` help kernel developer set the mask register that they need(will be used in compute_vector or simple-arithmetic-binaryop)

`void init_quantization` if your binary-alg contains `per-channel-quantization/dequantization`, you should call this function, passing a free `Reg64` and a free `Zmm`  before executing `compute_vector` to process the scale & zp.

`get_addr `will move the address in binaryop_attr to the register.

`compute_vector` the most important function. The behavior is to get the data stored in `src2`, if the bool value `broadcast ` is equal to true, then broadcast the scalar pointed by the `src2`, if equal to false then get the vector which begins address is `src2` and length equal to SIMD register's bit-width.<br />If `enable_mask` is equal to true, then before the computation results are stored to the `src1`, we apply the mask op and the mask register is set by set_mask function.Unlike `eltwise_injector`, `vector_compute` in `binaryop_injector` can only compute one binaryop_attr at a time. Users need to iterate through all bianryop_attr in a loop to apply all binaryop.
`add`/`sub`/`mul` is simple-arithmetic-binaryop function, the meaning of the params are similar to the `compute_vector`.`comput_vector` is the wrapper of the `add`/`sub`/`mul`.
```cpp
class jit_binary_injector {
 public:
  enum addr_type { normal, scale, zp };
  jit_binary_injector() {}
  virtual ~jit_binary_injector() {}
  void binary_injector_init(jit_generator* ptr);
  void set_mask(Opmask mask);
  void init_quantization(Zmm zmm, Reg64 reg);
  void get_addr(Reg64 reg, binaryop_attr op_attr,
                addr_type type = addr_type::normal);  // mov addr_ptr from op_attr to reg64.
  void compute_vector(Zmm zmm_src1, RegExp src2, binaryop_attr op_attr, bool enable_mask = false,
                      bool broadcast = false);
  void add(Zmm src1, RegExp src2, data_type op_dt, bool enable_mask, bool broadcast);
  void sub(Zmm src1, RegExp src2, data_type op_dt, bool enable_mask, bool broadcast);
  void mul(Zmm src1, RegExp src2, data_type op_dt, bool enable_mask, bool broadcast);
};
```
<a name="NTo8Z"></a>
## Usage
<a name="BVBDX"></a>
### Developer's Perspective
Transformers-accelerated Libraries developer only needs two steps to use the `binaryop_injector`.<br />step1. initial the `binaryop_injector` in kernel's construct function.
```cpp
binary_injector.binary_injector_init(this);
```
step2.config the address and mask(optional) and apply the binaryop which you want via `compute_vector `function. 
Note that users can also pass binaryop's address directly from their `data_param` to binaryop_addr `Reg64`.
```cpp
// get binaryop addr
binary_injector.get_addr(binaryop_addr, param_.binaryop_attrs.front());
// prepare RegExp
int append_loop_len = param_.process_col / (zmm_byte_size / get_data_size(param_.input_dt));
RegExp offset_exp = binaryop_addr + param_.thread_elt_offset * get_data_size(param_.input_dt) + (k % append_loop_len) * zmm_byte_size;
// apply binaryop
binary_injector.compute_vector(Zmm(k), offset_exp, param_.binaryop_attrs.front(), param_.input_dt);
```
<a name="tJ6bf"></a>
### User's Perspective
For the users of Transformers-accelerated Libraries, they only need to call `set_binaryop_list` to set the `bianryop_attr_list`.<br />And add a new key "binaryop_list" in `op_attrs` for kernel hashing.
```cpp
layernorm_ba_desc.set_binaryop_list({{append_vec, binaryop_alg::add}});
op_attrs["binaryop_list"]="binary_add";
```
