/*!
 * Copyright (c) 2022 by Contributors
 * \file constraint_opt.h
 * \brief Interface with the constraint optimization library.
 */
#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

// Define this macro to automatically find invalid constraints for debugging.
// #define CONSTRAINT_OPT_DEBUG

namespace operations_research {
namespace sat {
    class LinearExpr;
    class BoolVar;
    class IntVar;
    class CpModelBuilder;
}
}

namespace raf {
namespace plugins {
namespace constraint_opt {

enum class ConstraintType {
    kEq,
    kNeq,
    kAnd,
    kOr,
};

// direct c++ interface
class ConstraintOptimizer {
public:
    ConstraintOptimizer(int lower_bound);
    ConstraintOptimizer(const ConstraintOptimizer&);
    ~ConstraintOptimizer();
    int AddVariable(int upper_bound, const std::string& name = "");
    int AddConstant(int value, const std::string& name = "");
    int AddTarget(int bounded_var_idx, bool is_top, const std::string& name = "");
    int AddConstraint(ConstraintType constraint_type, int operands_0_idx, int operands_1_idx, bool is_top, const std::string& name = "");
    std::unordered_map<int, int> GetSolution();

private:
    int lower_bound_;
    std::unique_ptr<operations_research::sat::CpModelBuilder> cp_model_;
    std::vector<operations_research::sat::LinearExpr> elements_;
    std::vector<operations_research::sat::IntVar> variables_;
    std::vector<operations_research::sat::LinearExpr> targets_;
    std::unordered_map<int, int> variable_idx_to_elements_idx_;
#ifdef CONSTRAINT_OPT_DEBUG
    std::vector<operations_research::sat::BoolVar> enforcement_literals_;
#endif
};

#ifdef CONSTRAINT_OPT_JSON_INTERFACE
// interface using serialized json string
std::string SolveConstraintOpt(std::string operations_str);
#endif

} // namespace constraint_opt
} // namespace plugins
} // namespace raf