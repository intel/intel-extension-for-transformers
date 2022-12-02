# Customized Operators Register

It only takes two steps for developers to register a customized operator: 

[1. Add *.h of the customized operator to executor/include/operators](#1-add-h-of-the-customized-operator-to-executorincludeoperators)

[2. Add *.cpp of the customized operator to executor/src/operators](#2-add-cpp-of-the-customized-operator-to-executorsrcoperators)


Let's register a Gelu operator to engine executor as an example.

## 1. Add *.h of the customized operator to executor/include/operators
The *.h is a file to define member variables and functions. The GeluOperator class inherits from the Operator base class. The GeluOperator has a basic constructor, destructor, Reshape function and Forward function, and it also has some member variables used inside the class. 

The example uses oneDNN API, so we define some variables for onednn primitives. The details about oneDNN can refer the following link https://oneapi-src.github.io/oneDNN/index.html.
```cpp
class GeluOperator : public Operator {
 public:
  explicit GeluOperator(const shared_ptr<OperatorConfig>& conf);
  virtual ~GeluOperator() {}

  void Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) override;

 private:
  string algorithm_;
  dnnl::engine eng_ = engine(engine::kind::cpu, 0);
  dnnl::eltwise_forward gelu_p_;
  memory src_m_;
  memory dst_m_;
};
```
## 2. Add *.cpp of the customized operator to executor/src/operators
The following function is a constructor to parse the operator config. Gelu has two algorithm gelu_erf and gelu_tanh, and the attribute "algorithm" is parsed here.
```cpp
GeluOperator::GeluOperator(const shared_ptr<OperatorConfig>& conf) :
  Operator(conf) {
  auto attrs_map = operator_conf_->attributes();
  auto iter = attrs_map.find("algorithm");
  if (iter != attrs_map.end()) {
    algorithm_ = iter->second;
  }
}
```

The output shape can be dynamic, so we can adjust the shape of the output tensor in the reshape function. We also prepare primitives here, becuase the Gelu operator is based on oneDNN.
```cpp
void GeluOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  //// Part1: Prepare tensors shape and memory descriptors
  // 1.1: Prepare src tensor shape
  const memory::dims& src_shape = input[0]->shape();

  // 1.2 Set dst tensor shape
  const memory::dims& dst_shape = src_shape;
  Tensor* dst_tensor_ptr = output[0];
  dst_tensor_ptr->set_shape(dst_shape);

  // 1.3 Get tensor's strides
  memory::dims src_stride = GetStrides(src_shape);
  memory::dims dst_stride = GetStrides(dst_shape);

  // 1.4 Prepare memory descriptors
  memory::desc src_md(src_shape, memory::data_type::f32, src_stride);
  memory::desc dst_md(dst_shape, memory::data_type::f32, dst_stride);
  
  // 1.5 Prepare memory objects (cached)
  src_m_ = memory(src_md, eng_);
  dst_m_ = memory(dst_md, eng_);

  //// Part2: Prepare primitive
  // 2.1 Prepare op descriptors
  algorithm gelu_algorithm;
  if (algorithm_ == "gelu_erf") {
    gelu_algorithm = algorithm::eltwise_gelu_erf;
  } else if (algorithm_ == "gelu_tanh") {
    gelu_algorithm = algorithm::eltwise_gelu_tanh;
  } else {
    LOG(ERROR) << "Gelu algorithm is: " << algorithm_
               << ", not supported. Only gelu_erf or gelu_tanh is supported.";
  }
  auto gelu_d = dnnl::eltwise_forward::desc(prop_kind::forward_inference,
                gelu_algorithm, src_md, 0.f, 0.f);

  // 2.2 Prepare primitive descriptors
  dnnl::eltwise_forward::primitive_desc gelu_pd(gelu_d, eng_);

  // 2.3 Prepare primitive objects (cached)
  gelu_p_ = dnnl::eltwise_forward(gelu_pd);
}
```
The forward function executes the operator. Using oneDNN, we set input and output to data_handle, and execute the primitive. 

Notes: Please don't forget to unrefernce the input tensors after executing the primitive. The reason is to reduce the reference count of tensors and manage memory more strictly.

```cpp
void GeluOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
// 1. Alias variables part
const void* src_data = input[0]->data();
void* dst_data = output[0]->mutable_data();

// 2. Prepare memory objects with data_ptr
dnnl::stream s(eng_);
src_m_.set_data_handle(const_cast<void*>(src_data), s);
dst_m_.set_data_handle(reinterpret_cast<void*>(dst_data), s);

// 3. Insert memory args
memory_args_[DNNL_ARG_SRC] = src_m_;
memory_args_[DNNL_ARG_DST] = dst_m_;

// 4. Execute the primitive
gelu_p_.execute(s, memory_args_);

// 5. unref tensors
this->unref_tensors(input);
}
```

The prepare fusion is necessary for some kernels. Developers should set the dtype of output in prepare function especially when output is not fp32.
```cpp
void GeluOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  if (int8_lut_optimize || int8_lut_acc_test) output[0]->set_dtype("u8");
}
```

After creating the customized operator, finally register it to operator class as follow:
```
REGISTER_OPERATOR_CLASS(Gelu);
```

