/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/dialect/cuda/moe.cc
 * \brief moe cuda backend
 */
#include <cuda_runtime.h>
#include "raf/op.h"
#include "raf/pass.h"
#include "raf/op_utils.h"
#include "raf/device_api.h"
#include "raf/registry.h"
#include "raf/tensor.h"
#include "../tvm/tvm_fusion.h"
#include "../../schema/moe.h"
#include "../../../pass/let_list.h"
#include "raf/stream_pool.h"
#include "./kernels/kernel_util.cuh"

namespace raf {
namespace op {
namespace cuda {

using namespace tvm_dialect;
using tensor::Tensor;

TensorValue MakeConstTensor(Device to_dev, float value, DType dtype) {
  int64_t size = 1;
  std::vector<int64_t> shape = {};
  DLTensor tensor;
  if (dtype.bits == 32) {
    std::vector<float> a(size, value);
    tensor.data = a.data();
    tensor.device = Device(DevType::kCPU(), 0);
    tensor.dtype = dtype;
    tensor.shape = shape.data();
    tensor.ndim = 0;
    tensor.strides = nullptr;
    tensor.byte_offset = 0;
    auto array = tvm::runtime::NDArray::Empty(shape, dtype, to_dev);
    array.CopyFrom(&tensor);
    return TensorValue::make(Tensor::FromDLPack(array.ToDLPack()));
  } else {
    CHECK_EQ(dtype.bits, 16) << "Only support float32 or float16.";
    std::vector<half> a(size, value);
    tensor.data = a.data();
    tensor.device = Device(DevType::kCPU(), 0);
    tensor.dtype = dtype;
    tensor.shape = shape.data();
    tensor.ndim = 0;
    tensor.strides = nullptr;
    tensor.byte_offset = 0;
    auto array = tvm::runtime::NDArray::Empty(shape, dtype, to_dev);
    array.CopyFrom(&tensor);
    return TensorValue::make(Tensor::FromDLPack(array.ToDLPack()));
  }
}

Call MakeSumCall(Expr data, int axis) {
  static auto sum_op = Op::Get("raf.op.sum");
  std::vector<int64_t> axes_array = {axis};
  std::vector<int64_t> keepdims_array = {0};
  return Call(sum_op, {data,                                          // data
                       MakeConstant(ArrayToIntTuple(axes_array)),     // axes
                       MakeConstant(ArrayToIntTuple(keepdims_array)), // keepdims
                       MakeConstant(BoolValue::make(false))});        // exclude
}

using namespace raf::value;
using device_api::DeviceAPI;

Var MakeTypeCheckedVar(std::string name_hint, Type type_annotation) {
  auto var = raf::ir::MakeVar(name_hint, type_annotation);
  var->checked_type_ = type_annotation;
  return var;
}

inline std::vector<int64_t> GetShapeFromTensorValue(const Value& value) {
  ICHECK(value.defined());
  std::vector<int64_t> shape;
  if (const auto* tv = value.as<TensorValueObj>()) {
    DLTensor* tensor = GetRef<TensorValue>(tv);
    for (size_t i = 0; i < tensor->ndim; ++i) {
      shape.push_back(tensor->shape[i]);
    }
  } else {
    LOG(FATAL) << "Unsupported value type " << value;
  }
  return shape;
}

#define MoeEncodeInitBody \
    std::vector<int64_t> data_shape = GetShapeFromTensorValue(args->data); \
    std::vector<int64_t> gate_shape = GetShapeFromTensorValue(args->gate); \
 \
    CHECK_EQ(data_shape.size(), 2) << "Expected input data to be 2D."; \
    CHECK_EQ(gate_shape.size(), 2) << "Expected input data to be 2D."; \
 \
    CHECK_EQ(data_shape[0], gate_shape[0]) << "Input data and gate should have matching shape in dim 0."; \
    dim_S_ = data_shape[0]; \
    dim_M_ = data_shape[1]; \
    dim_E_ = gate_shape[1]; \
 \
    float capacity_factor = args->capacity_factor; \
 \
    dim_C_ = static_cast<int>(capacity_factor * ((dim_S_ + dim_E_ - 1) / dim_E_)); \
 \
    const DLTensor* data_tensor = args->data; \
    dev_ = data_tensor->device; \
 \
    CHECK_EQ(data_tensor->dtype.code, kDLFloat) << "Input data should be float."; \
 \
    if (data_tensor->dtype.bits == 32) { \
      data_dtype_ = DataType::Float(32); \
    } else { \
      CHECK_EQ(data_tensor->dtype.bits, 16) << "Input data should be float16 or float32."; \
      data_dtype_ = DataType::Float(16); \
    } \
    data_dl_dtype_ = data_tensor->dtype; \
 \
    Array<PrimExpr> se_shape = {Integer(dim_S_), Integer(dim_E_)}; \
    Array<PrimExpr> s_shape = {Integer(dim_S_)}; \
    masks_se_type_ = TensorType(se_shape, data_dtype_); \
    masks_s_float_type_ = TensorType(s_shape, data_dtype_); \
    masks_s_int_type_ = TensorType(s_shape, DataType::Int(32)); \
 \
    ones_ = MakeConstTensor(dev_, 1.0, data_tensor->dtype); \
    zeros_ = MakeConstTensor(dev_, 0.0, data_tensor->dtype); \
 \
    RequestWorkspace(&masks_se_buf_, dev_, dim_S_ * dim_E_ * (data_tensor->dtype.bits / 8)); \
 \
    auto argmax_func = GetArgMaxFunc(); \
    auto onehot_func = GetOneHotFunc(data_tensor->dtype.bits); \
    auto masked_gate_func = GetMaskedGatesFunc(); \
 \
    auto out_tuple = Downcast<TupleValue>(cv->out); \
    CHECK_EQ(out_tuple->fields.size(), 5); \
    auto argmax_tvm_func_cv = CallValues::make(); \
    Array<Value> argmax_tvm_func_args; \
    argmax_tvm_func_args.push_back(args->gate); \
    argmax_tvm_func_cv->args = MakeListArgs(argmax_tvm_func_args); \
    auto dummy_indices_output = value::CreateDummyValueFromType(masks_s_int_type_, dev_); \
    argmax_tvm_func_cv->out = dummy_indices_output; \
    argmax_tvm_func_cv->device = dev_; \
    argmax_func = Downcast<Function>(Cast2TVMDialect().Mutate(argmax_func)); \
    argmax_tvm_func_cv->callee = ClosureValue::make({}, argmax_func); \
    argmax_tvm_f_ = CompileFunc(argmax_tvm_func_cv); \
 \
    auto onehot_tvm_func_cv = CallValues::make(); \
    Array<Value> onehot_tvm_func_args; \
    onehot_tvm_func_args.push_back(out_tuple->fields[1]); \
    onehot_tvm_func_args.push_back(ones_); \
    onehot_tvm_func_args.push_back(zeros_); \
    onehot_tvm_func_cv->args = MakeListArgs(onehot_tvm_func_args); \
    auto dummy_onehot_output = value::CreateDummyValueFromType(masks_se_type_, dev_); \
    onehot_tvm_func_cv->out = dummy_onehot_output; \
    onehot_tvm_func_cv->device = dev_; \
    onehot_func = Downcast<Function>(Cast2TVMDialect().Mutate(onehot_func)); \
    onehot_tvm_func_cv->callee = ClosureValue::make({}, onehot_func); \
    onehot_tvm_f_ = CompileFunc(onehot_tvm_func_cv); \
 \
    auto masked_gate_tvm_func_cv = CallValues::make(); \
    Array<Value> masked_gate_tvm_func_args; \
    masked_gate_tvm_func_args.push_back(args->gate); \
    masked_gate_tvm_func_args.push_back(dummy_onehot_output); \
    masked_gate_tvm_func_cv->args = MakeListArgs(masked_gate_tvm_func_args); \
    masked_gate_tvm_func_cv->out = out_tuple->fields[0]; \
    masked_gate_tvm_func_cv->device = dev_; \
    masked_gate_func = Downcast<Function>(Cast2TVMDialect().Mutate(masked_gate_func)); \
    masked_gate_tvm_func_cv->callee = ClosureValue::make({}, masked_gate_func); \
    masked_gate_tvm_f_ = CompileFunc(masked_gate_tvm_func_cv);

#define MoeEncodeTVMFuncs \
  Function GetArgMaxFunc() { \
    static auto argmax_op = Op::Get("raf.op.argmax"); \
    auto gate_var = MakeTypeCheckedVar("moe_encode_gate", masks_se_type_); \
    LetList ll; \
    auto indices_s = ll.Push(Call(argmax_op, {gate_var, \
                                               MakeConstant(ScalarValue::make(1)), \
                                               MakeConstant(BoolValue::make(false)), \
                                               MakeConstant(BoolValue::make(false))})); \
    auto body = raf::pass::InferType(ll.Get(indices_s)); \
    Array<Var> params = {gate_var}; \
    auto func = Function(params, body, masks_s_int_type_, {}, {}, {}); \
    auto mod = IRModule::FromExpr(func); \
    mod = ToGraphNormalForm()(mod); \
    auto new_func = Downcast<Function>(mod->Lookup("main")); \
    Array<Type> param_types; \
    param_types.push_back(gate_var->checked_type_); \
    new_func->checked_type_ = FuncType(param_types, masks_s_int_type_, {}, {}); \
    return new_func; \
  } \
 \
  Function GetOneHotFunc(int input_bits) { \
    static auto one_hot_op = Op::Get("raf.op.one_hot"); \
    auto indices_s_var = MakeTypeCheckedVar("indices_s", masks_s_int_type_); \
    auto on_value_var = MakeTypeCheckedVar("on_value", TensorType({}, DataType::Float(input_bits))); \
    auto off_value_var = MakeTypeCheckedVar("off_value", TensorType({}, DataType::Float(input_bits))); \
    LetList ll; \
    std::string dtype_str; \
    if (input_bits == 32) { \
      dtype_str = "float32"; \
    } else { \
      CHECK_EQ(input_bits, 16) << "Unsupported input_bits " << input_bits; \
      dtype_str = "float16"; \
    } \
    auto masks_se = ll.Push(Call(one_hot_op, {indices_s_var, \
                                              on_value_var, \
                                              off_value_var, \
                                              MakeConstant(ScalarValue::make(dim_E_)), \
                                              MakeConstant(ScalarValue::make(1)), \
                                              MakeConstant(StringValue::make(dtype_str)), \
                                              MakeConstant(StringValue::make(std::string(dev_.c_str()))) \
                                            })); \
    auto body = raf::pass::InferType(ll.Get(masks_se)); \
    Array<Var> params = {indices_s_var, on_value_var, off_value_var}; \
    auto func = Function(params, body, masks_se_type_, {}, {}, {}); \
    auto mod = IRModule::FromExpr(func); \
    mod = ToGraphNormalForm()(mod); \
    auto new_func = Downcast<Function>(mod->Lookup("main")); \
    Array<Type> param_types; \
    param_types.push_back(indices_s_var->checked_type_); \
    param_types.push_back(on_value_var->checked_type_); \
    param_types.push_back(off_value_var->checked_type_); \
    new_func->checked_type_ = FuncType(param_types, masks_se_type_, {}, {}); \
    return new_func; \
  } \
 \
  Function GetMaskedGatesFunc() { \
    static auto multiply_op = Op::Get("raf.op.multiply"); \
    auto gate_var = MakeTypeCheckedVar("moe_encode_gate", masks_se_type_); \
    auto mask_var = MakeTypeCheckedVar("masks_se", masks_se_type_); \
    LetList ll; \
    auto masked_gates_se = ll.Push(Call(multiply_op, {gate_var, mask_var})); \
    auto masked_gates_s = ll.Push(MakeSumCall(masked_gates_se, 1)); \
    auto body = raf::pass::InferType(ll.Get(masked_gates_s)); \
    Array<Var> params = {gate_var, mask_var}; \
    auto func = Function(params, body, masks_s_float_type_, {}, {}, {}); \
    auto mod = IRModule::FromExpr(func); \
    mod = ToGraphNormalForm()(mod); \
    auto new_func = Downcast<Function>(mod->Lookup("main")); \
    Array<Type> param_types; \
    param_types.push_back(gate_var->checked_type_); \
    param_types.push_back(mask_var->checked_type_); \
    new_func->checked_type_ = FuncType(param_types, masks_s_float_type_, {}, {}); \
    return new_func; \
  }

#define MoeEncodeStates \
  tvm::DataType data_dtype_; \
  DLDataType data_dl_dtype_; \
 \
  registry::PackedFunc argmax_tvm_f_{nullptr}; \
  registry::PackedFunc onehot_tvm_f_{nullptr}; \
  registry::PackedFunc masked_gate_tvm_f_{nullptr}; \
 \
  TensorValue ones_; \
  TensorValue zeros_; \
 \
  void* masks_se_buf_; \
  Type masks_se_type_; \
  Type masks_s_float_type_; \
  Type masks_s_int_type_; \
 \
  Device dev_; \
 \
  int64_t dim_S_; \
  int64_t dim_E_; \
  int64_t dim_C_; \
  int64_t dim_M_;

#define MoeEncodePrepareAssignExpertsBody \
    std::vector<int64_t> se_buffer_shape = {dim_S_, dim_E_}; \
    value::TensorValue masks_se_buf_tensor = TensorValue::Assemble(dev_, data_dl_dtype_, se_buffer_shape, {}, masks_se_buf_); \
    DLTensor* masks_se_buf_dl_tensor = masks_se_buf_tensor; \
 \
    value::TupleValue out = ir::Downcast<TupleValue>(output); \
    CHECK_EQ(out->fields.size(), 5) << "Expected output to be a 5-Tuple, but only got " << out->fields.size() << " fields."; \
    DLTensor* gates_s = out->fields[0]; \
    DLTensor* indices_locations = out->fields[1]; \
    DLTensor* out_used_capacity = out->fields[2]; \
    DLTensor* out_elements_per_expert = out->fields[3]; \
    DLTensor* dispatched_input = out->fields[4]; \
    std::vector<int64_t> indices_1_s_shape = {dim_S_}; \
    DLTensor indices1_s = { \
      indices_locations->data, \
      indices_locations->device, \
      1, \
      indices_locations->dtype, \
      static_cast<int64_t*>(indices_1_s_shape.data()), \
      NULL, \
      indices_locations->byte_offset, \
    }; \
 \
    std::vector<DLTensor> argmax_tvm_inputs; \
    argmax_tvm_inputs.emplace_back(*gate); \
    std::vector<DLTensor> argmax_tvm_outputs; \
    argmax_tvm_outputs.emplace_back(indices1_s); \
    std::vector<TVMValue> argmax_values; \
    std::vector<int> argmax_codes; \
    SetArgs(&argmax_tvm_inputs, &argmax_tvm_outputs, &argmax_values, &argmax_codes); \
    TVMArgs argmax_targs(argmax_values.data(), argmax_codes.data(), argmax_values.size()); \
\
    DLTensor* ones = ones_; \
    DLTensor* zeros = zeros_; \
    std::vector<DLTensor> onehot_tvm_inputs; \
    onehot_tvm_inputs.emplace_back(indices1_s); \
    onehot_tvm_inputs.emplace_back(*ones); \
    onehot_tvm_inputs.emplace_back(*zeros); \
    std::vector<DLTensor> onehot_tvm_outputs; \
    onehot_tvm_outputs.emplace_back(*masks_se_buf_dl_tensor); \
    std::vector<TVMValue> onehot_values; \
    std::vector<int> onehot_codes; \
    SetArgs(&onehot_tvm_inputs, &onehot_tvm_outputs, &onehot_values, &onehot_codes); \
    TVMArgs onehot_targs(onehot_values.data(), onehot_codes.data(), onehot_values.size()); \
\
    std::vector<DLTensor> masked_gate_tvm_inputs; \
    masked_gate_tvm_inputs.emplace_back(*gate); \
    masked_gate_tvm_inputs.emplace_back(*masks_se_buf_dl_tensor); \
    std::vector<DLTensor> masked_gate_tvm_outputs; \
    masked_gate_tvm_outputs.emplace_back(*gates_s); \
    std::vector<TVMValue> masked_gate_values; \
    std::vector<int> masked_gate_codes; \
    SetArgs(&masked_gate_tvm_inputs, &masked_gate_tvm_outputs, &masked_gate_values, &masked_gate_codes); \
    TVMArgs masked_gate_targs(masked_gate_values.data(), masked_gate_codes.data(), masked_gate_values.size());

class MoeEncode : public raf::op::OpEnv {
 public:
  explicit MoeEncode(const CallValues& cv) {
    static auto fschema_index =
        ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    static auto op = ir::Op::Get("raf.op.moe_encode");
    auto args = cv->args.as<op::schema::MoeEncodeArgs>();
    MoeEncodeInitBody;

    this->arg_indices = {
        fschema_index[op]("data"),
        fschema_index[op]("gate"),
        fschema_index[op]("used_capacity")
    };
  }

  void Execute(const CallValues& cv) override {
    auto args = cv->args.as<op::schema::MoeEncodeArgs>();
    Execute(std::vector<value::Value>{args->data, args->gate, args->used_capacity}, cv->out);
  }

  void Execute(const std::vector<Value>& inputs, Value output) override {
    static auto cuda_device_api = DeviceAPI::Get(DevType::kCUDA());

    DLTensor* reshaped_input = Downcast<TensorValue>(inputs[0]);
    DLTensor* gate = Downcast<TensorValue>(inputs[1]);
    DLTensor* in_used_capacity = Downcast<TensorValue>(inputs[2]);

    MoeEncodePrepareAssignExpertsBody;

    // Skip the execution if we are in the task extraction mode since
    // we do not care about the correctness.
    if (AllowJitFailure()) {
      return;
    }

    TVMRetValue argmax_rv;
    argmax_tvm_f_.CallPacked(argmax_targs, &argmax_rv);
    TVMRetValue onehot_rv;
    onehot_tvm_f_.CallPacked(onehot_targs, &onehot_rv);
    TVMRetValue masked_gate_rv;
    masked_gate_tvm_f_.CallPacked(masked_gate_targs, &masked_gate_rv);

    int* locations1_s_data = static_cast<int*>(indices1_s.data) + dim_S_;
    // launch CUDA kernels
    launch_gen_location(static_cast<int*>(indices1_s.data),
                        static_cast<int*>(in_used_capacity->data),
                        locations1_s_data,
                        static_cast<int*>(out_used_capacity->data),
                        static_cast<uint64_t*>(out_elements_per_expert->data),
                        dim_C_, dim_S_, dim_E_, dim_M_, cuda_device_api->GetStream());
    if (data_dl_dtype_.bits == 32) {
      launch_encode_forward(static_cast<int*>(indices1_s.data),
                      locations1_s_data,
                      static_cast<int*>(in_used_capacity->data),
                      static_cast<float*>(reshaped_input->data),
                      static_cast<float*>(dispatched_input->data),
                      dim_S_, dim_M_, dim_C_, dim_E_, cuda_device_api->GetStream());
    } else {
      launch_encode_forward(static_cast<int*>(indices1_s.data),
                      locations1_s_data,
                      static_cast<int*>(in_used_capacity->data),
                      static_cast<half*>(reshaped_input->data),
                      static_cast<half*>(dispatched_input->data),
                      dim_S_, dim_M_, dim_C_, dim_E_, cuda_device_api->GetStream());
    }
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.cuda.moe_encode"));
  }

  static OpEnv* make(const CallValues& cv) {
    return new MoeEncode(cv);
  }

 protected:
  MoeEncodeTVMFuncs;
  MoeEncodeStates;
};

RAF_REGISTER_DIALECT_OP(cuda, moe_encode, 20);
RAF_OP_ENV_MAKER("raf.op.cuda.moe_encode", MoeEncode::make);


class MoeEncodeBatchPrioritized: public raf::op::OpEnv {
public:
  explicit MoeEncodeBatchPrioritized(const CallValues& cv) {
    static auto fschema_index =
        ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    static auto op = ir::Op::Get("raf.op.moe_encode");
    auto args = cv->args.as<op::schema::MoeEncodeBatchPrioritizedArgs>();
    std::vector<int64_t> data_shape = GetShapeFromTensorValue(args->data);
    std::vector<int64_t> gate_shape = GetShapeFromTensorValue(args->gate);

    n_partitions_ = args->n_partitions;
    partition_id_ = args->partition_id;

    CHECK_EQ(data_shape.size(), 2) << "Expected input data to be 2D.";
    CHECK_EQ(gate_shape.size(), 2) << "Expected input data to be 2D.";

    CHECK_EQ(data_shape[0], gate_shape[0]) << "Input data and gate should have matching shape in dim 0.";
    dim_orig_S_ = data_shape[0];
    dim_S_ = data_shape[0] / n_partitions_;
    dim_M_ = data_shape[1];
    dim_E_ = gate_shape[1];

    float capacity_factor = args->capacity_factor;
    dim_C_ = static_cast<int>(capacity_factor * ((dim_S_ + dim_E_ - 1) / dim_E_));

    const DLTensor* data_tensor = args->data;
    dev_ = data_tensor->device;

    CHECK_EQ(data_tensor->dtype.code, kDLFloat) << "Input data should be float.";

    if (data_tensor->dtype.bits == 32) {
      data_dtype_ = DataType::Float(32);
    } else {
      CHECK_EQ(data_tensor->dtype.bits, 16) << "Input data should be float16 or float32.";
      data_dtype_ = DataType::Float(16);
    }
    data_dl_dtype_ = data_tensor->dtype;

    Array<PrimExpr> se_shape = {Integer(dim_orig_S_), Integer(dim_E_)};
    Array<PrimExpr> s_shape = {Integer(dim_orig_S_)};
    masks_se_type_ = TensorType(se_shape, data_dtype_);
    masks_s_float_type_ = TensorType(s_shape, data_dtype_);
    masks_s_int_type_ = TensorType(s_shape, DataType::Int(32));

    ones_ = MakeConstTensor(dev_, 1.0, data_tensor->dtype);
    zeros_ = MakeConstTensor(dev_, 0.0, data_tensor->dtype);

    RequestWorkspace(&masks_se_buf_, dev_, dim_orig_S_ * dim_E_ * (data_tensor->dtype.bits / 8));

    auto argmax_func = GetArgMaxFunc();
    auto onehot_func = GetOneHotFunc(data_tensor->dtype.bits);
    auto masked_gate_func = GetMaskedGatesFunc();

    auto out_tuple = Downcast<TupleValue>(cv->out);
    CHECK_EQ(out_tuple->fields.size(), 5);
    auto argmax_tvm_func_cv = CallValues::make();
    Array<Value> argmax_tvm_func_args;
    argmax_tvm_func_args.push_back(args->gate);
    argmax_tvm_func_cv->args = MakeListArgs(argmax_tvm_func_args);
    auto dummy_indices_output = value::CreateDummyValueFromType(masks_s_int_type_, dev_);
    argmax_tvm_func_cv->out = dummy_indices_output;
    argmax_tvm_func_cv->device = dev_;
    argmax_func = Downcast<Function>(Cast2TVMDialect().Mutate(argmax_func));
    argmax_tvm_func_cv->callee = ClosureValue::make({}, argmax_func);
    argmax_tvm_f_ = CompileFunc(argmax_tvm_func_cv);

    auto onehot_tvm_func_cv = CallValues::make();
    Array<Value> onehot_tvm_func_args;
    onehot_tvm_func_args.push_back(dummy_indices_output);
    onehot_tvm_func_args.push_back(ones_);
    onehot_tvm_func_args.push_back(zeros_);
    onehot_tvm_func_cv->args = MakeListArgs(onehot_tvm_func_args);
    auto dummy_onehot_output = value::CreateDummyValueFromType(masks_se_type_, dev_);
    onehot_tvm_func_cv->out = dummy_onehot_output;
    onehot_tvm_func_cv->device = dev_;
    onehot_func = Downcast<Function>(Cast2TVMDialect().Mutate(onehot_func));
    onehot_tvm_func_cv->callee = ClosureValue::make({}, onehot_func);
    onehot_tvm_f_ = CompileFunc(onehot_tvm_func_cv);

    auto masked_gate_tvm_func_cv = CallValues::make();
    Array<Value> masked_gate_tvm_func_args;
    masked_gate_tvm_func_args.push_back(args->gate);
    masked_gate_tvm_func_args.push_back(dummy_onehot_output);
    masked_gate_tvm_func_cv->args = MakeListArgs(masked_gate_tvm_func_args);
    auto dummy_gate_s_output = value::CreateDummyValueFromType(masks_s_float_type_, dev_);
    masked_gate_tvm_func_cv->out = dummy_gate_s_output;
    masked_gate_tvm_func_cv->device = dev_;
    masked_gate_func = Downcast<Function>(Cast2TVMDialect().Mutate(masked_gate_func));
    masked_gate_tvm_func_cv->callee = ClosureValue::make({}, masked_gate_func);
    masked_gate_tvm_f_ = CompileFunc(masked_gate_tvm_func_cv);

    this->arg_indices = {
        fschema_index[op]("data"),
        fschema_index[op]("gate"),
    };
    // request additional workspace of size (dim_S_) and dtype int32
    // for calculating sorted indices and dropping mask
    RequestWorkspace(&sorted_indices_buffer_, dev_, dim_orig_S_ * sizeof(int32_t));
    // also request a sketchpad memory for dropping mask, used in generating location
    RequestWorkspace(&dropping_mask_buffer_, dev_, dim_orig_S_ * sizeof(bool));
    // since the output is of reduced size, we also need intermediate buffers
    // for the following:
    // indices_1s: dim_orig_S_
    RequestWorkspace(&indices_1s_buffer_, dev_, dim_orig_S_ * sizeof(int32_t));
    // gate_s: dim_orig_S_
    RequestWorkspace(&gate_s_buffer_, dev_, dim_orig_S_ * (data_tensor->dtype.bits / 8));
    // construct the tvm function
    auto argsort_func = GetArgSortFunc();
    // argsort function callvalue
    auto argsort_tvm_func_cv = CallValues::make();
    Array<Value> argsort_tvm_func_args;
    auto dummy_gate_s_input = value::CreateDummyValueFromType(masks_s_float_type_, dev_);
    argsort_tvm_func_args.push_back(dummy_gate_s_input);
    argsort_tvm_func_cv->args = MakeListArgs(argsort_tvm_func_args);
    // use a dummy value for compilation
    auto dummy_sorted_indices_output = value::CreateDummyValueFromType(masks_s_int_type_, dev_);
    argsort_tvm_func_cv->out = dummy_sorted_indices_output;
    argsort_tvm_func_cv->device = dev_;
    argsort_func = Downcast<Function>(Cast2TVMDialect().Mutate(argsort_func));
    argsort_tvm_func_cv->callee = ClosureValue::make({}, argsort_func);
    argsort_tvm_f_ = CompileFunc(argsort_tvm_func_cv);
  }

  void Execute(const CallValues& cv) override {
    auto args = cv->args.as<op::schema::MoeEncodeBatchPrioritizedArgs>();
    Execute(std::vector<value::Value>{args->data, args->gate}, cv->out);
  }

  void Execute(const std::vector<Value>& inputs, Value output) override {
    static auto cuda_device_api = DeviceAPI::Get(DevType::kCUDA());

    DLTensor* reshaped_input = Downcast<TensorValue>(inputs[0]);
    DLTensor* gate = Downcast<TensorValue>(inputs[1]);

    std::vector<int64_t> sorted_indices_buffer_shape = {dim_orig_S_};
    DLDataType sorted_indices_dtype;
    sorted_indices_dtype.code = kDLInt;
    sorted_indices_dtype.bits = 32;
    sorted_indices_dtype.lanes = 1;
    value::TensorValue sorted_indices_tensor = TensorValue::Assemble(dev_, sorted_indices_dtype, sorted_indices_buffer_shape, {}, sorted_indices_buffer_);
    DLTensor* sorted_indices_dl_tensor = sorted_indices_tensor;

    std::vector<int64_t> se_buffer_shape = {dim_orig_S_, dim_E_};
    value::TensorValue masks_se_buf_tensor = TensorValue::Assemble(dev_, data_dl_dtype_, se_buffer_shape, {}, masks_se_buf_);
    DLTensor* masks_se_buf_dl_tensor = masks_se_buf_tensor;

    value::TupleValue out = ir::Downcast<TupleValue>(output);
    CHECK_EQ(out->fields.size(), 5) << "Expected output to be a 5-Tuple, but only got " << out->fields.size() << " fields.";
    DLTensor* out_gates_s = out->fields[0];
    DLTensor* indices_locations = out->fields[1];
    DLTensor* out_used_capacity = out->fields[2];
    DLTensor* out_elements_per_expert = out->fields[3];
    DLTensor* dispatched_input = out->fields[4];
    // indices1_s is in intermediate buffer
    std::vector<int64_t> indices_1_s_shape = {dim_orig_S_};
    DLTensor indices1_s = {
      indices_1s_buffer_,
      indices_locations->device,
      1,
      indices_locations->dtype,
      static_cast<int64_t*>(indices_1_s_shape.data()),
      NULL,
      0,
    };
    // gate_s is in intermediate buffer
    std::vector<int64_t> gate_s_shape = {dim_orig_S_};
    DLTensor gates_s = {
      gate_s_buffer_,
      out_gates_s->device,
      1,
      out_gates_s->dtype,
      static_cast<int64_t*>(indices_1_s_shape.data()),
      NULL,
      0,
    };

    std::vector<DLTensor> argmax_tvm_inputs;
    argmax_tvm_inputs.emplace_back(*gate);
    std::vector<DLTensor> argmax_tvm_outputs;
    argmax_tvm_outputs.emplace_back(indices1_s);
    std::vector<TVMValue> argmax_values;
    std::vector<int> argmax_codes;
    SetArgs(&argmax_tvm_inputs, &argmax_tvm_outputs, &argmax_values, &argmax_codes);
    TVMArgs argmax_targs(argmax_values.data(), argmax_codes.data(), argmax_values.size());

    DLTensor* ones = ones_;
    DLTensor* zeros = zeros_;
    std::vector<DLTensor> onehot_tvm_inputs;
    onehot_tvm_inputs.emplace_back(indices1_s);
    onehot_tvm_inputs.emplace_back(*ones);
    onehot_tvm_inputs.emplace_back(*zeros);
    std::vector<DLTensor> onehot_tvm_outputs;
    onehot_tvm_outputs.emplace_back(*masks_se_buf_dl_tensor);
    std::vector<TVMValue> onehot_values;
    std::vector<int> onehot_codes;
    SetArgs(&onehot_tvm_inputs, &onehot_tvm_outputs, &onehot_values, &onehot_codes);
    TVMArgs onehot_targs(onehot_values.data(), onehot_codes.data(), onehot_values.size());

    std::vector<DLTensor> masked_gate_tvm_inputs;
    masked_gate_tvm_inputs.emplace_back(*gate);
    masked_gate_tvm_inputs.emplace_back(*masks_se_buf_dl_tensor);
    std::vector<DLTensor> masked_gate_tvm_outputs;
    masked_gate_tvm_outputs.emplace_back(gates_s);
    std::vector<TVMValue> masked_gate_values; \
    std::vector<int> masked_gate_codes; \
    SetArgs(&masked_gate_tvm_inputs, &masked_gate_tvm_outputs, &masked_gate_values, &masked_gate_codes); \
    TVMArgs masked_gate_targs(masked_gate_values.data(), masked_gate_codes.data(), masked_gate_values.size());
    // argsort
    std::vector<DLTensor> argsort_tvm_inputs;
    argsort_tvm_inputs.emplace_back(gates_s);
    std::vector<DLTensor> argsort_tvm_outputs;
    argsort_tvm_outputs.emplace_back(*sorted_indices_dl_tensor);
    std::vector<TVMValue> argsort_values;
    std::vector<int> argsort_codes;
    SetArgs(&argsort_tvm_inputs, &argsort_tvm_outputs, &argsort_values, &argsort_codes);
    TVMArgs argsort_targs(argsort_values.data(), argsort_codes.data(), argsort_values.size());

    // Skip the execution if we are in the task extraction mode since
    // we do not care about the correctness.
    if (AllowJitFailure()) {
      return;
    }

    TVMRetValue argmax_rv;
    argmax_tvm_f_.CallPacked(argmax_targs, &argmax_rv);
    TVMRetValue onehot_rv;
    onehot_tvm_f_.CallPacked(onehot_targs, &onehot_rv);
    TVMRetValue masked_gate_rv;
    masked_gate_tvm_f_.CallPacked(masked_gate_targs, &masked_gate_rv);
    TVMRetValue argsort_rv;
    argsort_tvm_f_.CallPacked(argsort_targs, &argsort_rv);

    int* out_indices1_s_data = static_cast<int*>(indices_locations->data);
    int* out_locations1_s_data = out_indices1_s_data + dim_S_;
    // we first copy gats_s from intermediate buffer to output
    void* gates_s_curr_partition = static_cast<int*>(gate_s_buffer_) + partition_id_ * dim_S_;
    CUDA_CALL(cudaMemcpyAsync(static_cast<void*>(out_gates_s->data), gates_s_curr_partition,
                              dim_S_ * data_dtype_.bits() / 8, cudaMemcpyDeviceToDevice, (cudaStream_t)cuda_device_api->GetStream()));
    // launch CUDA kernels
    launch_gen_location_bpr(static_cast<int*>(indices_1s_buffer_),
                            static_cast<int*>(sorted_indices_dl_tensor->data),
                            out_locations1_s_data,
                            static_cast<bool*>(dropping_mask_buffer_),
                            static_cast<int*>(out_used_capacity->data),
                            static_cast<uint64_t*>(out_elements_per_expert->data),
                            dim_C_, dim_orig_S_, dim_E_, dim_M_, 
                            n_partitions_, partition_id_,
                            cuda_device_api->GetStream());
    // now we copy the corresponding indices_s from intermediate buffer to output
    void* indices_1s_curr_partition = static_cast<int*>(indices_1s_buffer_) + partition_id_ * dim_S_;
    CUDA_CALL(cudaMemcpyAsync(static_cast<void*>(out_indices1_s_data), indices_1s_curr_partition,
                              dim_S_ * sizeof(int32_t), cudaMemcpyDeviceToDevice, (cudaStream_t)cuda_device_api->GetStream()));

    // feed in offseted reshaped_input
    if (data_dl_dtype_.bits == 32) {
      float* offseted_reshaped_input = static_cast<float*>(reshaped_input->data) + partition_id_ * dim_S_ * dim_M_;
      launch_encode_forward(static_cast<int*>(out_indices1_s_data),
                      out_locations1_s_data,
                      offseted_reshaped_input,
                      static_cast<float*>(dispatched_input->data),
                      dim_S_, dim_M_, dim_C_, dim_E_, cuda_device_api->GetStream());
    } else {
      half* offseted_reshaped_input = static_cast<half*>(reshaped_input->data) + partition_id_ * dim_S_ * dim_M_;
      launch_encode_forward(static_cast<int*>(out_indices1_s_data),
                      out_locations1_s_data,
                      offseted_reshaped_input,
                      static_cast<half*>(dispatched_input->data),
                      dim_S_, dim_M_, dim_C_, dim_E_, cuda_device_api->GetStream());
    }
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.cuda.moe_encode_batch_prioritized"));
  }

  static OpEnv* make(const CallValues& cv) {
    return new MoeEncodeBatchPrioritized(cv);
  }

protected:
  MoeEncodeTVMFuncs;

  int64_t n_partitions_;
  int64_t partition_id_;
  int64_t dim_orig_S_;

  Function GetArgSortFunc() {
    static auto argsort_op = Op::Get("raf.op.argsort");
    auto gate_var = MakeTypeCheckedVar("gate_s", masks_s_float_type_);
    LetList ll;
    auto indices_s = ll.Push(Call(argsort_op, {gate_var,
                                               MakeConstant(ScalarValue::make(0)),
                                               MakeConstant(BoolValue::make(false))}));
    auto body = raf::pass::InferType(ll.Get(indices_s));
    Array<Var> params = {gate_var};
    auto func = Function(params, body, masks_s_int_type_, {}, {}, {});
    // Convert the function into GNF
    auto mod = IRModule::FromExpr(func);
    mod = ToGraphNormalForm()(mod);
    auto new_func = Downcast<Function>(mod->Lookup("main"));
    Array<Type> param_types;
    param_types.push_back(gate_var->checked_type_);
    new_func->checked_type_ = FuncType(param_types, masks_s_int_type_, {}, {});
    return new_func;
  }
  MoeEncodeStates;
  void* sorted_indices_buffer_ = nullptr;
  void* dropping_mask_buffer_ = nullptr;
  void* indices_1s_buffer_ = nullptr;
  void* gate_s_buffer_ = nullptr;
  registry::PackedFunc argsort_tvm_f_{nullptr};
};

RAF_REGISTER_DIALECT_OP(cuda, moe_encode_batch_prioritized, 20);
RAF_OP_ENV_MAKER("raf.op.cuda.moe_encode_batch_prioritized", MoeEncodeBatchPrioritized::make);

class MoeMergeMasks : public raf::op::OpEnv {
 public:
  explicit MoeMergeMasks(const CallValues& cv) {
    static auto fschema_index =
        ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    static auto op = ir::Op::Get("raf.op.moe_merge_masks");
    auto args = cv->args.as<op::schema::MoeMergeMasksArgs>();

    n_partitions_ = args->indices_locations.size();

    std::vector<int64_t> mask_shape = GetShapeFromTensorValue(args->indices_locations[0]);    // [2, S]

    CHECK_EQ(mask_shape.size(), 2) << "Expected input indices_locations to be 2D.";

    dim_S_ = mask_shape[1];
    dim_E_ = args->n_experts;

    const DLTensor* mask_tensor0 = args->indices_locations[0];
    dev_ = mask_tensor0->device;

    // request device memory for storing the array of ptrs
    // 2 -> one for indices and one for locations
    RequestWorkspace(&dev_ptr_buffer, cv->device, sizeof(void*) * n_partitions_ * 2);
    // We also allocate pinned memory on the host for faster data transfer
    CUDA_CALL(cudaMallocHost(&host_ptr_buffer, sizeof(void*) * n_partitions_ * 2));

    this->arg_indices = {
        fschema_index[op]("indices_locations"),
    };
  }

  ~MoeMergeMasks() {
    CUDA_CALL(cudaFreeHost(host_ptr_buffer));
  }

  void Execute(const CallValues& cv) override {
    auto args = cv->args.as<op::schema::MoeMergeMasksArgs>();
    Execute({TupleValue::make(ir::Array<Value>{args->indices_locations.begin(), args->indices_locations.end()})},
            cv->out);
  }

  void Execute(const std::vector<Value>& inputs, Value output) override {
    static auto cuda_device_api = DeviceAPI::Get(DevType::kCUDA());

    auto in_indices_locations = Downcast<TupleValue>(inputs[0]);

    CHECK_EQ(in_indices_locations->fields.size(), n_partitions_);
    // fill in the host buffer
    void** host_in_indices_locations_ptrs = static_cast<void**>(host_ptr_buffer);
    for (int i = 0; i < n_partitions_; ++i) {
      void* in_dev_ptr = static_cast<DLTensor*>(Downcast<TensorValue>(in_indices_locations->fields[i]))->data;
      // indices
      *(host_in_indices_locations_ptrs + i) = in_dev_ptr;
      // locations
      *(host_in_indices_locations_ptrs + i + n_partitions_) = static_cast<int*>(in_dev_ptr) + dim_S_;
    }
    // copy the host buffer to device
    CUDA_CALL(cudaMemcpyAsync(dev_ptr_buffer, host_ptr_buffer, sizeof(void*) * n_partitions_ * 2,
                              cudaMemcpyHostToDevice, (cudaStream_t)cuda_device_api->GetStream()));

    void** dev_in_indices_ptrs = static_cast<void**>(dev_ptr_buffer);
    void** dev_in_locations_ptrs = dev_in_indices_ptrs + n_partitions_;

    int* recon_indices = static_cast<int*>(static_cast<DLTensor*>(ir::Downcast<TensorValue>(output))->data);
    int* recon_locations = recon_indices + dim_S_ * n_partitions_;

    launch_merge_masks(reinterpret_cast<int**>(dev_in_indices_ptrs),
                      reinterpret_cast<int**>(dev_in_locations_ptrs),
                      recon_indices,
                      recon_locations,
                      dim_S_, dim_E_, n_partitions_, cuda_device_api->GetStream());
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.cuda.moe_merge_masks"));
  }

  static OpEnv* make(const CallValues& cv) {
    return new MoeMergeMasks(cv);
  }

 private:
  Device dev_;
  void* dev_ptr_buffer;
  void* host_ptr_buffer;

  int64_t dim_S_;
  int64_t dim_E_;
  int64_t n_partitions_;
};

RAF_REGISTER_DIALECT_OP(cuda, moe_merge_masks, 20);
RAF_OP_ENV_MAKER("raf.op.cuda.moe_merge_masks", MoeMergeMasks::make);


class MoeRedispatch : public raf::op::OpEnv {
 public:
  explicit MoeRedispatch(const CallValues& cv) {
    static auto fschema_index =
        ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    static auto op = ir::Op::Get("raf.op.moe_redispatch");
    auto args = cv->args.as<op::schema::MoeRedispatchArgs>();

    n_partitions_ = args->indices_locations.size();

    std::vector<int64_t> mask_shape = GetShapeFromTensorValue(args->indices_locations[0]);    // [2, S]
    std::vector<int64_t> data_shape = GetShapeFromTensorValue(args->data[0]);

    CHECK_EQ(mask_shape.size(), 2) << "Expected input indices_locations to be 2D.";

    dim_S_ = mask_shape[1];
    if (data_shape.size() == 3) {
      // [E, C, M]
      dim_E_ = data_shape[0];
      dim_C_ = data_shape[1];
      dim_M_ = data_shape[2];
    } else {
      CHECK_EQ(data_shape.size(), 4) << "Expected input data to be 3 or 4D.";
      // [G, LE, C, M]
      dim_E_ = data_shape[0] * data_shape[1];
      dim_C_ = data_shape[2];
      dim_M_ = data_shape[3];
    }

    const DLTensor* mask_tensor0 = args->indices_locations[0];
    dev_ = mask_tensor0->device;

    // request device memory for storing the array of ptrs
    // 2 -> one for data, one for indices
    RequestWorkspace(&dev_ptr_buffer, cv->device, sizeof(void*) * n_partitions_ * 2);
    // We also allocate pinned memory on the host for faster data transfer
    CUDA_CALL(cudaMallocHost(&host_ptr_buffer, sizeof(void*) * n_partitions_ * 2));

    this->arg_indices = {
      fschema_index[op]("data"),
      fschema_index[op]("indices_locations"),
    };
  }

  ~MoeRedispatch() {
    CUDA_CALL(cudaFreeHost(host_ptr_buffer));
  }

  void Execute(const CallValues& cv) override {
    auto args = cv->args.as<op::schema::MoeRedispatchArgs>();
    Execute({TupleValue::make(ir::Array<Value>{args->data.begin(), args->data.end()}),
            TupleValue::make(ir::Array<Value>{args->indices_locations.begin(), args->indices_locations.end()})},
            cv->out);
  }

  void Execute(const std::vector<Value>& inputs, Value output) override {
    static auto cuda_device_api = DeviceAPI::Get(DevType::kCUDA());

    auto in_data = Downcast<TupleValue>(inputs[0]);
    auto in_indices_locations = Downcast<TupleValue>(inputs[1]);

    CHECK_EQ(in_indices_locations->fields.size(), n_partitions_);
    // fill in the host buffer
    void** host_in_data_ptrs = static_cast<void**>(host_ptr_buffer);
    void** host_in_indices_ptrs = host_in_data_ptrs + n_partitions_;
    for (int i = 0; i < n_partitions_; ++i) {
      void* in_data_ptr = static_cast<DLTensor*>(Downcast<TensorValue>(in_data->fields[i]))->data;
      void* in_inc_loc_ptr = static_cast<DLTensor*>(Downcast<TensorValue>(in_indices_locations->fields[i]))->data;
      // data
      *(host_in_data_ptrs + i) = in_data_ptr;
      // indices
      *(host_in_indices_ptrs + i) = in_inc_loc_ptr;
    }
    // copy the host buffer to device
    CUDA_CALL(cudaMemcpyAsync(dev_ptr_buffer, host_ptr_buffer, sizeof(void*) * n_partitions_ * 2,
                              cudaMemcpyHostToDevice, (cudaStream_t)cuda_device_api->GetStream()));

    void** dev_in_data_ptrs = static_cast<void**>(dev_ptr_buffer);
    void** dev_in_indices_ptrs = dev_in_data_ptrs + n_partitions_;

    DLTensor* out_data = static_cast<DLTensor*>(Downcast<TensorValue>(output));
    CHECK_EQ(out_data->dtype.code, kDLFloat) << "Expected output data to be float.";
    if (out_data->dtype.bits == 32) {
      launch_redispatch(reinterpret_cast<float**>(dev_in_data_ptrs),
                        reinterpret_cast<int**>(dev_in_indices_ptrs),
                        static_cast<float*>(out_data->data),
                        dim_S_, dim_M_, dim_C_, dim_E_, n_partitions_, cuda_device_api->GetStream());
    } else {
      CHECK_EQ(out_data->dtype.bits, 16) << "Expected output data to be float32 or float16.";
      launch_redispatch(reinterpret_cast<half**>(dev_in_data_ptrs),
                        reinterpret_cast<int**>(dev_in_indices_ptrs),
                        static_cast<half*>(out_data->data),
                        dim_S_, dim_M_, dim_C_, dim_E_, n_partitions_, cuda_device_api->GetStream());
    }
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.cuda.moe_redispatch"));
  }

  static OpEnv* make(const CallValues& cv) {
    return new MoeRedispatch(cv);
  }

 private:
  Device dev_;
  void* dev_ptr_buffer;
  void* host_ptr_buffer;

  int64_t dim_S_;
  int64_t dim_E_;
  int64_t dim_C_;
  int64_t dim_M_;
  int64_t n_partitions_;
};

RAF_REGISTER_DIALECT_OP(cuda, moe_redispatch, 20);
RAF_OP_ENV_MAKER("raf.op.cuda.moe_redispatch", MoeRedispatch::make);

class MoeRedispatchExpertInput : public raf::op::OpEnv {
 public:
  explicit MoeRedispatchExpertInput(const CallValues& cv) {
    static auto fschema_index =
        ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    static auto op = ir::Op::Get("raf.op.moe_redispatch_expert_input");
    auto args = cv->args.as<op::schema::MoeRedispatchExpertInputArgs>();

    n_partitions_ = args->recv_cnts.size();

    std::vector<int64_t> mask_shape = GetShapeFromTensorValue(args->recv_cnts[0]);    // [G x LE]
    std::vector<int64_t> data_shape = GetShapeFromTensorValue(args->data[0]);

    dim_LE_ = args->n_local_experts;
    recv_cnts_scale_ = args->recv_cnts_scale;

    CHECK_EQ(mask_shape.size(), 1) << "Expected input indices_locations to be 2D.";

    if (data_shape.size() == 4) {
      // [C, 1, G, M]
      dim_G_ = data_shape[2];
      dim_C_ = data_shape[0];
      dim_M_ = data_shape[3];
    } else if (data_shape.size() == 3) {
      // [C, G, M]
      dim_G_ = data_shape[1];
      dim_C_ = data_shape[0];
      dim_M_ = data_shape[2];
    } else {
      // [C x G, M]
      CHECK_EQ(data_shape.size(), 2) << "Expected input data to be 2, 3 or 4D.";
      dim_G_ = mask_shape[0] / dim_LE_;
      dim_C_ = data_shape[0] / dim_G_;
      dim_M_ = data_shape[1];
    }
    CHECK_EQ(mask_shape[0] / dim_G_, dim_LE_) << "Expected input recv_cnts to be 1D of size G x LE. But got mask_shape[0]: " << mask_shape[0] << ", dim_G_: " << dim_G_ << ", dim_LE_: " << dim_LE_;
    local_expert_id_ = args->local_expert_id;

    const DLTensor* data_tensor0 = args->data[0];
    dev_ = data_tensor0->device;

    // request device memory for storing the array of ptrs
    // 2 -> one for data, one for masks
    RequestWorkspace(&dev_ptr_buffer, cv->device, sizeof(void*) * n_partitions_ * 2);
    // We also allocate pinned memory on the host for faster data transfer
    CUDA_CALL(cudaMallocHost(&host_ptr_buffer, sizeof(void*) * n_partitions_ * 2));

    this->arg_indices = {
      fschema_index[op]("data"),
      fschema_index[op]("recv_cnts"),
    };
  }

  ~MoeRedispatchExpertInput() {
    CUDA_CALL(cudaFreeHost(host_ptr_buffer));
  }

  void Execute(const CallValues& cv) override {
    auto args = cv->args.as<op::schema::MoeRedispatchExpertInputArgs>();
    Execute({TupleValue::make(ir::Array<Value>{args->data.begin(), args->data.end()}),
            TupleValue::make(ir::Array<Value>{args->recv_cnts.begin(), args->recv_cnts.end()})},
            cv->out);
  }

  void Execute(const std::vector<Value>& inputs, Value output) override {
    static auto cuda_device_api = DeviceAPI::Get(DevType::kCUDA());

    auto in_data = Downcast<TupleValue>(inputs[0]);
    auto in_recv_cnts = Downcast<TupleValue>(inputs[1]);

    CHECK_EQ(in_recv_cnts->fields.size(), n_partitions_);
    // fill in the host buffer
    void** host_in_data_ptrs = static_cast<void**>(host_ptr_buffer);
    void** host_in_masks_ptrs = host_in_data_ptrs + n_partitions_;
    for (int i = 0; i < n_partitions_; ++i) {
      void* in_data_ptr = static_cast<DLTensor*>(Downcast<TensorValue>(in_data->fields[i]))->data;
      void* in_mask_ptr = static_cast<DLTensor*>(Downcast<TensorValue>(in_recv_cnts->fields[i]))->data;
      // data
      *(host_in_data_ptrs + i) = in_data_ptr;
      // indices
      *(host_in_masks_ptrs + i) = in_mask_ptr;
    }
    // copy the host buffer to device
    CUDA_CALL(cudaMemcpyAsync(dev_ptr_buffer, host_ptr_buffer, sizeof(void*) * n_partitions_ * 2,
                              cudaMemcpyHostToDevice, (cudaStream_t)cuda_device_api->GetStream()));

    void** dev_in_data_ptrs = static_cast<void**>(dev_ptr_buffer);
    void** dev_in_mask_ptrs = dev_in_data_ptrs + n_partitions_;

    DLTensor* out_data = static_cast<DLTensor*>(Downcast<TensorValue>(output));
    CHECK_EQ(out_data->dtype.code, kDLFloat) << "Expected output data to be float.";
    if (out_data->dtype.bits == 32) {
      launch_redispatch_expert_input(reinterpret_cast<float**>(dev_in_data_ptrs),
                        reinterpret_cast<uint64_t**>(dev_in_mask_ptrs),
                        static_cast<float*>(out_data->data), local_expert_id_,
                        dim_LE_, dim_C_, dim_M_, dim_G_, n_partitions_, recv_cnts_scale_, cuda_device_api->GetStream());
    } else {
      CHECK_EQ(out_data->dtype.bits, 16) << "Expected output data to be float32 or float16.";
      launch_redispatch_expert_input(reinterpret_cast<half**>(dev_in_data_ptrs),
                        reinterpret_cast<uint64_t**>(dev_in_mask_ptrs),
                        static_cast<half*>(out_data->data), local_expert_id_,
                        dim_LE_, dim_C_, dim_M_, dim_G_, n_partitions_, recv_cnts_scale_, cuda_device_api->GetStream());
    }
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.cuda.moe_redispatch_expert_input"));
  }

  static OpEnv* make(const CallValues& cv) {
    return new MoeRedispatchExpertInput(cv);
  }

 private:
  Device dev_;
  void* dev_ptr_buffer;
  void* host_ptr_buffer;

  int64_t dim_LE_;
  int64_t dim_C_;
  int64_t dim_M_;
  int64_t dim_G_;
  int64_t local_expert_id_;
  int64_t n_partitions_;
  int64_t recv_cnts_scale_;
};

RAF_REGISTER_DIALECT_OP(cuda, moe_redispatch_expert_input, 20);
RAF_OP_ENV_MAKER("raf.op.cuda.moe_redispatch_expert_input", MoeRedispatchExpertInput::make);

class MoeEncodeDx : public raf::op::OpEnv {
 public:
  explicit MoeEncodeDx(const CallValues& cv) {
    static auto fschema_index =
        ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    static auto op = ir::Op::Get("raf.op.moe_encode_dx");
    auto args = cv->args.as<op::schema::MoeEncodeDxArgs>();

    std::vector<int64_t> dy_shape = GetShapeFromTensorValue(args->dy);                                // [E, C, M]
    std::vector<int64_t> indices_locations_shape = GetShapeFromTensorValue(args->indices_locations);  // [2, S]

    CHECK_EQ(dy_shape.size(), 3) << "Expected input dy to be 3D.";
    CHECK_EQ(indices_locations_shape.size(), 2) << "Expected input indices_locations_shape to be 2D.";

    dim_S_ = indices_locations_shape[1];
    dim_E_ = dy_shape[0];
    dim_C_ = dy_shape[1];
    dim_M_ = dy_shape[2];

    const DLTensor* dy_tensor = args->dy;
    dev_ = dy_tensor->device;

    this->arg_indices = {
        fschema_index[op]("dy"),
        fschema_index[op]("indices_locations"),
    };
  }

  void Execute(const CallValues& cv) override {
    auto args = cv->args.as<op::schema::MoeEncodeDxArgs>();
    Execute(std::vector<value::Value>{args->dy, args->indices_locations}, cv->out);
  }

  void Execute(const std::vector<Value>& inputs, Value output) override {
    static auto cuda_device_api = DeviceAPI::Get(DevType::kCUDA());

    DLTensor* dy_dispatched_input = Downcast<TensorValue>(inputs[0]);
    DLTensor* indices_locations = Downcast<TensorValue>(inputs[1]);

    int* indices_ptr = static_cast<int*>(indices_locations->data);
    int* locations_ptr = indices_ptr + dim_S_;

    DLTensor* out = ir::Downcast<TensorValue>(output);
    CHECK_EQ(out->dtype.code, kDLFloat) << "Expected output data to be float.";
    if (out->dtype.bits == 32) {
      launch_encode_backward_data(indices_ptr,
                                  locations_ptr,
                                  static_cast<float*>(out->data),
                                  static_cast<float*>(dy_dispatched_input->data), 
                                  dim_S_, dim_M_, dim_C_, dim_E_, cuda_device_api->GetStream());
    } else {
      CHECK_EQ(out->dtype.bits, 16) << "Expected output data to be float32 or float16.";
      launch_encode_backward_data(indices_ptr,
                                  locations_ptr,
                                  static_cast<half*>(out->data),
                                  static_cast<half*>(dy_dispatched_input->data),
                                  dim_S_, dim_M_, dim_C_, dim_E_, cuda_device_api->GetStream());
    }
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.cuda.moe_encode_dx"));
  }

  static OpEnv* make(const CallValues& cv) {
    return new MoeEncodeDx(cv);
  }

 private:
  Device dev_;

  int64_t dim_S_;
  int64_t dim_E_;
  int64_t dim_C_;
  int64_t dim_M_;
};

RAF_REGISTER_DIALECT_OP(cuda, moe_encode_dx, 20);
RAF_OP_ENV_MAKER("raf.op.cuda.moe_encode_dx", MoeEncodeDx::make);

class MoeEncodeDg : public raf::op::OpEnv {
 public:
  explicit MoeEncodeDg(const CallValues& cv) {
    static auto fschema_index =
        ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    static auto op = ir::Op::Get("raf.op.moe_encode_dg");
    auto args = cv->args.as<op::schema::MoeEncodeDgArgs>();

    std::vector<int64_t> dy_shape = GetShapeFromTensorValue(args->dy);      // [S]
    std::vector<int64_t> mask_shape = GetShapeFromTensorValue(args->indices_locations);  // [2, S]

    CHECK_EQ(dy_shape.size(), 1) << "Expected input dy to be 1D.";
    CHECK_EQ(mask_shape.size(), 2) << "Expected input indices_locations to be 2D.";

    dim_S_ = dy_shape[0];
    dim_E_ = args->n_experts;

    const DLTensor* dy_tensor = args->dy;
    dev_ = dy_tensor->device;

    this->arg_indices = {
        fschema_index[op]("dy"),
        fschema_index[op]("indices_locations"),
    };
  }

  void Execute(const CallValues& cv) override {
    auto args = cv->args.as<op::schema::MoeEncodeDgArgs>();
    Execute(std::vector<value::Value>{args->dy, args->indices_locations}, cv->out);
  }

  void Execute(const std::vector<Value>& inputs, Value output) override {
    static auto cuda_device_api = DeviceAPI::Get(DevType::kCUDA());

    DLTensor* dy_gates1_s = Downcast<TensorValue>(inputs[0]);
    DLTensor* indices_locations = Downcast<TensorValue>(inputs[1]);


    DLTensor* out = ir::Downcast<TensorValue>(output);
    CHECK_EQ(out->dtype.code, kDLFloat) << "Expected output data to be float.";
    if (out->dtype.bits == 32) {
      launch_encode_backward_gate(static_cast<float*>(dy_gates1_s->data),
                                static_cast<int*>(indices_locations->data),
                                static_cast<float*>(out->data),
                                dim_S_, dim_E_, cuda_device_api->GetStream());
    } else {
      CHECK_EQ(out->dtype.bits, 16) << "Expected output data to be float32 or float16.";
      launch_encode_backward_gate(static_cast<half*>(dy_gates1_s->data),
                                static_cast<int*>(indices_locations->data),
                                static_cast<half*>(out->data),
                                dim_S_, dim_E_, cuda_device_api->GetStream());
    }
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.cuda.moe_encode_dg"));
  }

  static OpEnv* make(const CallValues& cv) {
    return new MoeEncodeDg(cv);
  }

 private:
  Device dev_;

  int64_t dim_S_;
  int64_t dim_E_;
};

RAF_REGISTER_DIALECT_OP(cuda, moe_encode_dg, 20);
RAF_OP_ENV_MAKER("raf.op.cuda.moe_encode_dg", MoeEncodeDg::make);


class MoeDecode : public raf::op::OpEnv {
 public:
  explicit MoeDecode(const CallValues& cv) {
    static auto fschema_index =
        ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    static auto op = ir::Op::Get("raf.op.moe_decode");
    auto args = cv->args.as<op::schema::MoeDecodeArgs>();

    std::vector<int64_t> data_shape = GetShapeFromTensorValue(args->data);    // [E, C, M]
    std::vector<int64_t> gate_shape = GetShapeFromTensorValue(args->gate);    // [S]

    CHECK_EQ(data_shape.size(), 3) << "Expected input data to be 3D.";
    CHECK_EQ(gate_shape.size(), 1) << "Expected input gate to be 1D.";

    dim_S_ = gate_shape[0];
    dim_E_ = data_shape[0];
    dim_C_ = data_shape[1];
    dim_M_ = data_shape[2];

    const DLTensor* data_tensor = args->data;
    dev_ = data_tensor->device;

    this->arg_indices = {
        fschema_index[op]("data"),
        fschema_index[op]("gate"),
        fschema_index[op]("indices_locations"),
    };
  }

  void Execute(const CallValues& cv) override {
    auto args = cv->args.as<op::schema::MoeDecodeArgs>();
    Execute(std::vector<value::Value>{args->data, args->gate, args->indices_locations}, cv->out);
  }

  void Execute(const std::vector<Value>& inputs, Value output) override {
    static auto cuda_device_api = DeviceAPI::Get(DevType::kCUDA());

    DLTensor* expert_output = Downcast<TensorValue>(inputs[0]);
    DLTensor* gates_s = Downcast<TensorValue>(inputs[1]);
    DLTensor* indices_locations = Downcast<TensorValue>(inputs[2]);

    DLTensor* out = ir::Downcast<TensorValue>(output);

    int* indices1_s = static_cast<int*>(indices_locations->data);
    int* locations1_s = indices1_s + dim_S_;

    CHECK_EQ(out->dtype.code, kDLFloat) << "Expected output data to be float.";
    if (out->dtype.bits == 32) {
      launch_decode_forward(static_cast<float*>(gates_s->data),
                          indices1_s,
                          locations1_s,
                          static_cast<float*>(out->data),
                          static_cast<float*>(expert_output->data), 
                          dim_S_, dim_M_, dim_C_, dim_E_, cuda_device_api->GetStream());
    } else {
      CHECK_EQ(out->dtype.bits, 16) << "Expected output data to be float32 or float16.";
      launch_decode_forward(static_cast<half*>(gates_s->data),
                          indices1_s,
                          locations1_s,
                          static_cast<half*>(out->data),
                          static_cast<half*>(expert_output->data), 
                          dim_S_, dim_M_, dim_C_, dim_E_, cuda_device_api->GetStream());
    }
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.cuda.moe_decode"));
  }

  static OpEnv* make(const CallValues& cv) {
    return new MoeDecode(cv);
  }

 private:
  Device dev_;

  int64_t dim_S_;
  int64_t dim_E_;
  int64_t dim_C_;
  int64_t dim_M_;
};

RAF_REGISTER_DIALECT_OP(cuda, moe_decode, 20);
RAF_OP_ENV_MAKER("raf.op.cuda.moe_decode", MoeDecode::make);

class MoeDecodeDx : public raf::op::OpEnv {
 public:
  explicit MoeDecodeDx(const CallValues& cv) {
    static auto fschema_index =
        ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    static auto op = ir::Op::Get("raf.op.moe_decode_dx");
    auto args = cv->args.as<op::schema::MoeDecodeDxArgs>();

    std::vector<int64_t> dy_shape = GetShapeFromTensorValue(args->dy);        // [S, M]

    CHECK_EQ(dy_shape.size(), 2) << "Expected input dy to be 2D.";

    dim_S_ = dy_shape[0];
    dim_E_ = args->n_experts;
    dim_C_ = args->capacity;
    dim_M_ = dy_shape[1];

    const DLTensor* dy_tensor = args->dy;
    dev_ = dy_tensor->device;

    this->arg_indices = {
        fschema_index[op]("dy"),
        fschema_index[op]("gate"),
        fschema_index[op]("indices_locations"),
    };
  }

  void Execute(const CallValues& cv) override {
    auto args = cv->args.as<op::schema::MoeDecodeDxArgs>();
    Execute(std::vector<value::Value>{args->dy, args->gate, args->indices_locations}, cv->out);
  }

  void Execute(const std::vector<Value>& inputs, Value output) override {
    static auto cuda_device_api = DeviceAPI::Get(DevType::kCUDA());

    DLTensor* dy_combined_output = Downcast<TensorValue>(inputs[0]);
    DLTensor* gates_s = Downcast<TensorValue>(inputs[1]);
    DLTensor* indices_locations = Downcast<TensorValue>(inputs[2]);

    DLTensor* out = ir::Downcast<TensorValue>(output);

    int* indices1_s = static_cast<int*>(indices_locations->data);
    int* locations1_s = indices1_s + dim_S_;

    CHECK_EQ(out->dtype.code, kDLFloat) << "Expected output data to be float.";

    if(out->dtype.bits == 32) {
      launch_decode_backward_data(static_cast<float*>(gates_s->data),
                          indices1_s,
                          locations1_s,
                          static_cast<float*>(dy_combined_output->data),
                          static_cast<float*>(out->data),
                          dim_S_, dim_M_, dim_C_, dim_E_, cuda_device_api->GetStream());
    } else {
      CHECK_EQ(out->dtype.bits, 16) << "Expected output data to be float32 or float16.";
      launch_decode_backward_data(static_cast<half*>(gates_s->data),
                          indices1_s,
                          locations1_s,
                          static_cast<half*>(dy_combined_output->data),
                          static_cast<half*>(out->data),
                          dim_S_, dim_M_, dim_C_, dim_E_, cuda_device_api->GetStream());
    }
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.cuda.moe_decode_dx"));
  }

  static OpEnv* make(const CallValues& cv) {
    return new MoeDecodeDx(cv);
  }

 private:
  Device dev_;

  int64_t dim_S_;
  int64_t dim_E_;
  int64_t dim_C_;
  int64_t dim_M_;
};

RAF_REGISTER_DIALECT_OP(cuda, moe_decode_dx, 20);
RAF_OP_ENV_MAKER("raf.op.cuda.moe_decode_dx", MoeDecodeDx::make);

class MoeDecodeDg : public raf::op::OpEnv {
 public:
  explicit MoeDecodeDg(const CallValues& cv) {
    static auto fschema_index =
        ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    static auto op = ir::Op::Get("raf.op.moe_decode_dg");
    auto args = cv->args.as<op::schema::MoeDecodeDgArgs>();

    std::vector<int64_t> data_shape = GetShapeFromTensorValue(args->data);    // [E, C, M]
    std::vector<int64_t> dy_shape = GetShapeFromTensorValue(args->dy);        // [S, M]

    CHECK_EQ(data_shape.size(), 3) << "Expected input data to be 3D.";
    CHECK_EQ(dy_shape.size(), 2) << "Expected input dy to be 2D.";

    dim_S_ = dy_shape[0];
    dim_E_ = data_shape[0];
    dim_C_ = data_shape[1];
    dim_M_ = data_shape[2];

    const DLTensor* data_tensor = args->data;
    dev_ = data_tensor->device;

    this->arg_indices = {
        fschema_index[op]("dy"),
        fschema_index[op]("data"),
        fschema_index[op]("indices_locations"),
    };
  }

  void Execute(const CallValues& cv) override {
    auto args = cv->args.as<op::schema::MoeDecodeDgArgs>();
    Execute(std::vector<value::Value>{args->dy, args->data, args->indices_locations}, cv->out);
  }

  void Execute(const std::vector<Value>& inputs, Value output) override {
    static auto cuda_device_api = DeviceAPI::Get(DevType::kCUDA());

    DLTensor* dy_combined_output = Downcast<TensorValue>(inputs[0]);
    DLTensor* expert_output = Downcast<TensorValue>(inputs[1]);
    DLTensor* indices_locations = Downcast<TensorValue>(inputs[2]);

    DLTensor* out = ir::Downcast<TensorValue>(output);

    int* indices1_s = static_cast<int*>(indices_locations->data);
    int* locations1_s = indices1_s + dim_S_;

    CHECK_EQ(out->dtype.code, kDLFloat) << "Expected output data to be float.";

    if (out->dtype.bits == 32) {
      launch_decode_backward_gate(static_cast<float*>(out->data),
                          indices1_s,
                          locations1_s,
                          static_cast<float*>(dy_combined_output->data),
                          static_cast<float*>(expert_output->data),
                          dim_S_, dim_M_, dim_C_, cuda_device_api->GetStream());
    } else {
      CHECK_EQ(out->dtype.bits, 16) << "Expected output data to be float32 or float16.";
      launch_decode_backward_gate(static_cast<half*>(out->data),
                          indices1_s,
                          locations1_s,
                          static_cast<half*>(dy_combined_output->data),
                          static_cast<half*>(expert_output->data),
                          dim_S_, dim_M_, dim_C_, cuda_device_api->GetStream());
    }
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.cuda.moe_decode_dg"));
  }

  static OpEnv* make(const CallValues& cv) {
    return new MoeDecodeDg(cv);
  }

 private:
  Device dev_;

  int64_t dim_S_;
  int64_t dim_E_;
  int64_t dim_C_;
  int64_t dim_M_;
};

RAF_REGISTER_DIALECT_OP(cuda, moe_decode_dg, 20);
RAF_OP_ENV_MAKER("raf.op.cuda.moe_decode_dg", MoeDecodeDg::make);

}  // namespace cuda
}  // namespace op
}  // namespace raf
