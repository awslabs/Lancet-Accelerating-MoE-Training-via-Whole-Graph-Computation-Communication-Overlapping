/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file constraint_opt.cc
 * \brief Cost models used in schedule simulator.
 */
#include <stdlib.h>
#include <thread>
#ifdef CONSTRAINT_OPT_JSON_INTERFACE
#include "json.hpp"
#endif
#include "raf/plugins/ortools/constraint_opt.h"
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model.pb.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/util/sorted_interval_list.h"


namespace raf {
namespace plugins {
namespace constraint_opt {

#ifdef CONSTRAINT_OPT_JSON_INTERFACE
using namespace nlohmann;
#endif
using operations_research::sat::CpSolverResponse;
using operations_research::sat::CpSolverStatus;
using operations_research::sat::CpModelBuilder;
using operations_research::sat::LinearExpr;
using operations_research::sat::BoolVar;
using operations_research::sat::IntVar;
using operations_research::sat::Solve;
using operations_research::sat::SolveWithParameters;
using operations_research::sat::SatParameters;

inline bool inString(std::string where, std::string target) {
    // find if target is in where
    return where.find(target) != std::string::npos;
}

void checkConstraintComponent(std::string var_name) {
    // constraint component can be constraint, target and constant
    CHECK(inString(var_name, "constraint") ||
          inString(var_name, "target") ||
          inString(var_name, "constant"));
}

int getVariableIndex(std::string name) {
    auto loc_underscore = name.find("_");
    CHECK(loc_underscore != std::string::npos);
    int start_idx = loc_underscore + 1;
    auto loc_vbar = name.find("|");
    CHECK(loc_vbar != std::string::npos);
    int end_idx = loc_vbar;
    CHECK_GT(end_idx, start_idx);
    return std::stoi(name.substr(start_idx, end_idx - start_idx));
}

operations_research::sat::LinearExpr Sum(std::vector<operations_research::sat::LinearExpr> exprs) {
    CHECK_GT(exprs.size(), 0);
    operations_research::sat::LinearExpr result = exprs[0];
    for(int i=1; i<exprs.size(); i++) {
        result += exprs[i];
    }
    return result;
}

ConstraintOptimizer::ConstraintOptimizer(int lower_bound): cp_model_(new CpModelBuilder()), lower_bound_(lower_bound) {}
ConstraintOptimizer::ConstraintOptimizer(const ConstraintOptimizer& other) : cp_model_(new CpModelBuilder(*other.cp_model_)){
    lower_bound_ = other.lower_bound_;
    elements_ = other.elements_;
    variables_ = other.variables_;
    targets_ = other.targets_;
    variable_idx_to_elements_idx_ = other.variable_idx_to_elements_idx_;
#ifdef CONSTRAINT_OPT_DEBUG
    enforcement_literals_ = other.enforcement_literals_;
#endif
}
ConstraintOptimizer::~ConstraintOptimizer() {}

int ConstraintOptimizer::AddVariable(int upper_bound, const std::string& name) {
    int op_idx = elements_.size();
    auto variable = cp_model_->NewIntVar({lower_bound_, upper_bound});
    if (name != "") {
        variable = variable.WithName("variable_" + std::to_string(op_idx) + "|" + name);
    }
    variable_idx_to_elements_idx_[variables_.size()] = op_idx;
    elements_.push_back(variable);
    variables_.push_back(variable);
    return op_idx;
}

int ConstraintOptimizer::AddConstant(int value, const std::string& name) {
    int op_idx = elements_.size();
    auto constant = cp_model_->NewIntVar({value, value});
    if (name != "") {
        constant = constant.WithName("constant_" + std::to_string(op_idx) + "|" + name);
    }
    elements_.push_back(constant);
    return op_idx;
}

int ConstraintOptimizer::AddTarget(int bounded_var_idx, bool is_top, const std::string& name) {
    int op_idx = elements_.size();
    auto bounded_var = elements_[bounded_var_idx];
    if(is_top) {
        targets_.push_back(bounded_var);
    }
    return bounded_var_idx;
}

int ConstraintOptimizer::AddConstraint(ConstraintType constraint_type, int operands_0_idx, int operands_1_idx, bool is_top, const std::string& name) {
    int op_idx = elements_.size();
    auto constraint = cp_model_->NewBoolVar();
    if (name != "") {
        auto constraint_name = "constraint_" + std::to_string(op_idx) + "|" + name;
        if(is_top) {
            constraint_name = "top_" + constraint_name;
        }
        constraint = constraint.WithName(constraint_name);
    }
    elements_.push_back(constraint);
    auto var_1 = elements_[operands_0_idx];
    auto var_2 = elements_[operands_1_idx];
    if(constraint_type == ConstraintType::kEq) {
        cp_model_->AddEquality(var_1, var_2).OnlyEnforceIf(constraint);
        cp_model_->AddNotEqual(var_1, var_2).OnlyEnforceIf(Not(constraint));
    } else if (constraint_type == ConstraintType::kNeq) {
        cp_model_->AddNotEqual(var_1, var_2).OnlyEnforceIf(constraint);
        cp_model_->AddEquality(var_1, var_2).OnlyEnforceIf(Not(constraint));
    } else if (constraint_type == ConstraintType::kAnd) {
        // checkConstraintComponent(var_1.Name());
        // checkConstraintComponent(var_2.Name());
        std::vector<LinearExpr> var1and2 = {var_1, var_2};
        auto sum_var1_var2 = Sum(var1and2);
        cp_model_->AddEquality(sum_var1_var2, 2).OnlyEnforceIf(constraint);
        cp_model_->AddNotEqual(sum_var1_var2, 2).OnlyEnforceIf(Not(constraint));
    } else if (constraint_type == ConstraintType::kOr) {
        // checkConstraintComponent(var_1.Name());
        // checkConstraintComponent(var_2.Name());
        std::vector<LinearExpr> var1and2 = {var_1, var_2};
        auto sum_var1_var2 = Sum(var1and2);
        cp_model_->AddGreaterThan(sum_var1_var2, 0).OnlyEnforceIf(constraint);
        cp_model_->AddLessOrEqual(sum_var1_var2, 0).OnlyEnforceIf(Not(constraint));
    }
    if(is_top) {
#ifdef CONSTRAINT_OPT_DEBUG
        BoolVar enforcement_literal = cp_model_->NewBoolVar();
        enforcement_literal = enforcement_literal.WithName("enforcement_literal_" + std::to_string(op_idx) + "|" + name);
        enforcement_literals_.push_back(enforcement_literal);
        cp_model_->AddEquality(constraint, 1).OnlyEnforceIf(enforcement_literal);
#else
        cp_model_->AddEquality(constraint, 1);
#endif
    }
    return op_idx;
}

std::unordered_map<int, int> ConstraintOptimizer::GetSolution() {
    SatParameters parameters;
    parameters.set_num_search_workers(std::max<int>(std::thread::hardware_concurrency(), 1));
#ifdef CONSTRAINT_OPT_DEBUG
    cp_model_->AddAssumptions(enforcement_literals_);
#endif
    if(targets_.size() > 0) {
        LinearExpr target_sum = targets_[0];
        for(int i=1; i<targets_.size(); i++) {
            target_sum += targets_[i];
        }
        cp_model_->Maximize(target_sum);
    }
    const CpSolverResponse response = Solve(cp_model_->Build());
    if (response.status() != CpSolverStatus::OPTIMAL &&
        response.status() != CpSolverStatus::FEASIBLE) {
#ifdef CONSTRAINT_OPT_DEBUG
        for (auto index: response.sufficient_assumptions_for_infeasibility()) {
            LOG(INFO) << "Infeasible assumption: " << cp_model_->GetBoolVarFromProtoIndex(index).Name();
        }
#endif
        throw std::runtime_error("Constraint mode error: No solution found:" + std::to_string(response.status()));
    }
    std::unordered_map<int, int> solution;
    for(int var_idx=0; var_idx < variables_.size(); var_idx++) {
        auto variable = variables_[var_idx];
        auto variable_name = variable.Name();
        auto variable_value = SolutionIntegerValue(response, variable);
        int element_idx = variable_idx_to_elements_idx_.at(var_idx);
        solution[element_idx] = variable_value;
    }
    return solution;
}

#ifdef CONSTRAINT_OPT_JSON_INTERFACE
// interface using serialized json string
std::string SolveConstraintOpt(std::string operations_str) {
    CpModelBuilder cp_model;
    auto operations_json = json::parse(operations_str);

    std::vector<LinearExpr> elements;
    std::vector<IntVar> variables;
    std::vector<BoolVar> targets;

    // std::unordered_map<int, IntVar> intvar_map;
    // std::unordered_map<int, IntVar> constant_map;
    // std::unordered_map<int, BoolVar> boolvar_map;
    // std::unordered_map<int, Constraint> constraint_map;
    // std::unordered_map<int, BoolVar> target_map;

    for(int idx =0; idx < operations_json.size(); idx ++) {
        auto operation = operations_json[idx];
        auto name = operation["name"].get<std::string>();
        if(operation["type"] == "variable") {
            auto upper_bound = operation["upper_bound"];
            auto variable = cp_model.NewIntVar({0, upper_bound}).WithName("variable_" + std::to_string(idx) + "|" + name);
            elements.push_back(variable);
            variables.push_back(variable);
            // intvar_map[idx] = variable;
        } else if(operation["type"] == "constant") {
            auto value = operation["value"];
            auto constant = cp_model.NewIntVar({value, value}).WithName("constant_" + std::to_string(idx) + "|" + name);
            elements.push_back(constant);
            // constant_map[idx] = constant;
        } else if(operation["type"] == "target") {
            auto target = cp_model.NewBoolVar().WithName("target_" + std::to_string(idx) + "|" + name);
            elements.push_back(target);
            auto bounded_var = elements[operation["bounded_var"]];
            cp_model.AddGreaterThan(bounded_var, 0).OnlyEnforceIf(target);
            cp_model.AddLessOrEqual(bounded_var, 0).OnlyEnforceIf(Not(target));
            if(operation["is_top"]) {
                targets.push_back(target);
            }
        } else if(operation["type"] == "constraint") {
            auto op = operation["op"];
            auto constraint_name = "constraint_" + std::to_string(idx) + "|" + name;
            if(operation["is_top"]) {
                constraint_name = "top_" + constraint_name;
            }
            auto constraint = cp_model.NewBoolVar().WithName(constraint_name);
            elements.push_back(constraint);
            std::vector<int> operands = operation["operands"];
            CHECK_EQ(operands.size(), 2);
            int idx_1 = operands[0];
            int idx_2 = operands[1];
            auto var_1 = elements[idx_1];
            auto var_2 = elements[idx_2];
            if(op == "eq") {
                cp_model.AddEquality(var_1, var_2).OnlyEnforceIf(constraint);
                cp_model.AddNotEqual(var_1, var_2).OnlyEnforceIf(Not(constraint));
            } else if (op == "neq") {
                cp_model.AddNotEqual(var_1, var_2).OnlyEnforceIf(constraint);
                cp_model.AddEquality(var_1, var_2).OnlyEnforceIf(Not(constraint));
            } else if (op == "and") {
                // checkConstraintComponent(var_1.Name());
                // checkConstraintComponent(var_2.Name());
                std::vector<LinearExpr> var1and2 = {var_1, var_2};
                auto sum_var1_var2 = Sum(var1and2);
                cp_model.AddEquality(sum_var1_var2, 2).OnlyEnforceIf(constraint);
                cp_model.AddNotEqual(sum_var1_var2, 2).OnlyEnforceIf(Not(constraint));
            } else if (op == "or") {
                // checkConstraintComponent(var_1.Name());
                // checkConstraintComponent(var_2.Name());
                std::vector<LinearExpr> var1and2 = {var_1, var_2};
                auto sum_var1_var2 = Sum(var1and2);
                cp_model.AddGreaterThan(sum_var1_var2, 0).OnlyEnforceIf(constraint);
                cp_model.AddLessOrEqual(sum_var1_var2, 0).OnlyEnforceIf(Not(constraint));
            }
            if(operation["is_top"]) {
                cp_model.AddEquality(constraint, 1);
            }
        }
    }

    cp_model.Maximize(LinearExpr::Sum(targets));
    const CpSolverResponse response = Solve(cp_model.Build());
    if (response.status() != CpSolverStatus::OPTIMAL &&
        response.status() != CpSolverStatus::FEASIBLE) {
        // LOG(FATAL) << "Constraint mode error: No solution found.";
        throw std::runtime_error("Constraint mode error: No solution found.");
    }
    std::string result = "";
    for(auto variable: variables) {
        result += "\n" + std::to_string(getVariableIndex(variable.Name())) + " " + std::to_string(SolutionIntegerValue(response, variable));
    }
    if(result.size()) {
        result = result.substr(1);
    }
    return result;
}
#endif

}  // namespace constraint_opt
}  // namespace plugins
}  // namespace raf