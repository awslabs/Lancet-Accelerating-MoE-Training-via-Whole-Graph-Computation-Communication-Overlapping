/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file ./src/op/dialect/tvm/tvm_utils.h
 * \brief Implementation of utility methods for TVM dialect.
 */
#include "raf/value.h"
#include "raf/registry.h"
#include "raf/op.h"
#include "raf/ir.h"
#include "raf/ir_ext.h"
#include "raf/pass.h"
#include "tvm/ir/type_functor.h"
#include "tvm/auto_scheduler/compute_dag.h"
#include "tvm/runtime/packed_func.h"
#include "relay/backend/te_compiler.h"
#include "relay/backend/te_compiler_cache.h"
#include "./tvm_utils.h"
#include "../../../op/schema/list_args.h"
#include "../../../common/shape_utils.h"

namespace raf {
namespace op {
namespace tvm_dialect {

using namespace raf::value;
using namespace raf::op;
using namespace raf::pass;
using namespace raf::ir;
using namespace tvm::runtime;

/*! \brief Cast base and dialect ops to TVM dialect if possible. */
class Cast2TVMDialect : public ExprMutator {
 public:
  Expr VisitExpr(const Expr& expr) override;
  Expr VisitExpr_(const OpNode* node) override;
};

/*!
 * \brief Converter from raf style (all inputs are arguments) to
 *        tvm style (inputs are explicitly marked as arguments or attrs)
 */
class Meta2TVM : public ExprMutator {
 public:
  Meta2TVM(const CallValues& call, const DevType& dev_type);

  Expr operator()();
  Expr VisitExpr(const Expr& expr) override;
  Expr VisitExpr_(const VarNode* node) override;
  Expr VisitExpr_(const CallNode* node) override;
  Expr VisitExpr_(const FunctionNode* node) override;

 public:
  /*! \brief the indices of fused function params that correspond to tvm non-attr */
  std::vector<int> arg_indices;
  /*! \brief readable function name */
  std::string func_name;

 private:
  /*! \brief convert CallNode to CallValues */
  CallValuesGetter call_values_getter_;
  /*! \brief params that are tvm op inputs, instead of attrs */
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> input_;
  /*! \brief the primitive function to be analyzed */
  Function func_;
  /*! \brief The device type */
  DevType device_type_;
};

PackedFunc CompileFunc(const CallValues& call);

OpEnv* FusedFuncBuild(const CallValues& call);

float CalcFuncGFLOPS(const CallValues& call, const Array<Type>& param_types,
                     const Type& ret_type, const Device& device);

} // namespace tvm_dialect
} // namespace op
} // namespace raf