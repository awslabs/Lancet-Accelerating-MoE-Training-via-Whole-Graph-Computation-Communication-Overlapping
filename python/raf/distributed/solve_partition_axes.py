# pylint: disable=protected-access, invalid-name
"""Util functions related to partitioning"""
import json
from ortools.sat.python import cp_model

from raf._lib import tvm

def check_constraint_component(var):
    # constraint component can be constraint, target and constant
    assert "constraint" in var.Name() or "target" in var.Name() or "constant" in var.Name()

def check_variable(var):
    assert "variable" in var.Name()

def check_constant(var):
    assert "constant" in var.Name()

def get_variable_idx(var):
    name = var.Name()
    start_idx = name.find("_") + 1
    end_idx = name.find("|")
    assert not (start_idx == 0 or end_idx == -1)
    return int(name[start_idx:end_idx])


@tvm._ffi.register_func("raf.distributed.solve_partition_axes")
def solve_partition_axes(operations_str):
    """
    Parameters
    ----------
    operations_str : str
        Operations of creating variables, adding constraints on variables
        and setting targets in JSON format.

    Returns
    -------
    result : str
        Solver status and values of all variables in an optimal or feasible solution.
    """
    model = cp_model.CpModel()
    elements = []
    variables = []
    targets = []
    operations = json.loads(operations_str)
    for idx, operation in enumerate(operations):
        name = operation["name"]
        if operation["type"] == "variable":
            upper_bound = operation["upper_bound"]
            variable = model.NewIntVar(0, upper_bound, "variable_{}|{}".format(idx, name))
            elements.append(variable)
            variables.append(variable)
        elif operation["type"] == "constant":
            value = operation["value"]
            constant = model.NewIntVar(value, value, "constant_{}|{}".format(idx, name))
            elements.append(constant)
        elif operation["type"] == "target":
            target = model.NewBoolVar("target_{}|{}".format(idx, name))
            elements.append(target)
            bounded_var = elements[operation["bounded_var"]]
            model.Add(bounded_var > 0).OnlyEnforceIf(target)
            model.Add(bounded_var <= 0).OnlyEnforceIf(target.Not())

            if operation["is_top"]:
                targets.append(target)

        elif operation["type"] == "constraint":
            op = operation["op"]
            constraint_name = "constraint_{}|{}".format(idx, name)
            if operation["is_top"]:
                constraint_name = "top_" + constraint_name
            constraint = model.NewBoolVar(constraint_name)
            elements.append(constraint)
            idx_1, idx_2 = operation["operands"]
            var_1, var_2 = elements[idx_1], elements[idx_2]
            if op == "eq":
                model.Add(var_1 == var_2).OnlyEnforceIf(constraint)
                model.Add(var_1 != var_2).OnlyEnforceIf(constraint.Not())
            elif op == "neq":
                model.Add(var_1 != var_2).OnlyEnforceIf(constraint)
                model.Add(var_1 == var_2).OnlyEnforceIf(constraint.Not())
            elif op == "and":
                check_constraint_component(var_1)
                check_constraint_component(var_2)
                model.Add(sum([var_1, var_2]) == 2).OnlyEnforceIf(constraint)
                model.Add(sum([var_1, var_2]) != 2).OnlyEnforceIf(constraint.Not())
            elif op == "or":
                check_constraint_component(var_1)
                check_constraint_component(var_2)
                model.Add(sum([var_1, var_2]) > 0).OnlyEnforceIf(constraint)
                model.Add(sum([var_1, var_2]) <= 0).OnlyEnforceIf(constraint.Not())

            if operation["is_top"]:
                model.Add(constraint == 1)

    model.Maximize(sum(targets))
    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    status = solver.Solve(model)
    # print("Status: {}, maximium of objective function: {}, targets count: {} ".format(
    #     solver.StatusName(status), solver.ObjectiveValue(), len(targets)))
    result = ""
    for variable in variables:
        result += "\n" + str(get_variable_idx(variable)) + " " + str(solver.Value(variable))
    result = result[1:]
    return result
