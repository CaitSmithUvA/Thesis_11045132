################################################################################
# step3.py
# 29/01/2021
# Caitlin Smith
# 11045132
#
# Universiteit van Amsterdam
# Bachelor Scriptie: Approximating the Hamiltonian of the double pendulum using
# an evolutionary algorithm
#
# Step 4 of the proposed structure. Calculates the derivative pairings and eval-
# uates these as proposed by Schmidt and Lipson
################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import operator
import math
import re

from sympy import *
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations, implicit_multiplication_application)
from sympy.parsing.mathematica import mathematica
from sympy.parsing.latex import parse_latex

import step3 as s3

# calculateing the individual time derivatives
def calc_time_derivatives(list_x, list_y, list_t):
    dx_dt = [list_x[0] - 1]
    dy_dt = [list_y[0] - 1]

    # exploiting that all lists have same dimension
    for i in range(1, len(list_x)):
        dx = list_x[i] - list_x[i - 1]
        dy = list_y[i] - list_y[i - 1]
        dt = list_t[i] - list_t[i - 1]

        dx_dt.append(dx/dt)
        dy_dt.append(dy/dt)

    return dx_dt, dy_dt

# calculating the derivative pairs in time sequence according to SOM S1
def time_derivative(list_a, list_b):
    return [i / j for i, j in zip(list_a, list_b)]

# create all derivative pairings neccessary for the double pendulum
def time_derivatives(theta_1, theta_2, omega_1, omega_2, time_double):
    dp1_dt, do1_dt = calc_time_derivatives(theta_1, omega_1, time_double)
    t_p1o1_div = time_derivative(dp1_dt, do1_dt)
    dp1_dt, do2_dt = calc_time_derivatives(theta_1, omega_2, time_double)
    t_p1o2_div = time_derivative(dp1_dt, do2_dt)
    dp2_dt, do1_dt = calc_time_derivatives(theta_2, omega_1, time_double)
    t_p2o1_div = time_derivative(dp2_dt, do1_dt)
    dp2_dt, do2_dt = calc_time_derivatives(theta_2, omega_2, time_double)
    t_p2o2_div = time_derivative(dp2_dt, do2_dt)
    return t_p1o1_div, t_p1o2_div, t_p2o1_div, t_p2o2_div


# calculate derivatives of two input functions
def func_derivative(func, derivative1, derivative2):
    f_d1 = func.diff(derivative1)
    f_d2 = func.diff(derivative2)
    return f_d1, f_d2

# calculate derivative values for two individual variables at each timestep
def calc_derivatives(f_d1, f_d2, var_lists, SYSTEM):
    df_d1 = []
    df_d2 = []
    # choose right input form for system
    for i in range(len(var_lists[0])):
        if SYSTEM == "DP":
            f_1 = f_d1.evalf(4, subs = dict(a = var_lists[0][i], b = var_lists[1][i], e = var_lists[2][i], f = var_lists[3][i]))
            f_2 = f_d2.evalf(4, subs = dict(a = var_lists[0][i], b = var_lists[1][i], e = var_lists[2][i], f = var_lists[3][i]))
        elif SYSTEM == "SP_AB":
            f_1 = f_d1.evalf(4, subs = dict(a = var_lists[0][i], b = var_lists[1][i]))
            f_2 = f_d2.evalf(4, subs = dict(a = var_lists[0][i], b = var_lists[1][i]))
        else:
            f_1 = f_d1.evalf(4, subs = dict(x = var_lists[0][i], y = var_lists[1][i]))
            f_2 = f_d2.evalf(4, subs = dict(x = var_lists[0][i], y = var_lists[1][i]))
        df_d1.append(f_1)
        df_d2.append(f_2)
    return df_d1, df_d2

# calculate dx/dy by performing (df/dy)/(df/dx)
def indiv_derivative(df_d1, df_d2):
    d_df = []
    count = 0
    for i in range(len(df_d1)):
        if df_d1[i] != 0:
            eval = df_d2[i]/df_d1[i]
            if type(eval) != "float":
                d_df.append(df_d2[i]/df_d1[i])
            else:
                d_df.append(round(df_d2[i]/df_d1[i], 4))
        else:
            count+=1
            d_df.append(0)
    return d_df

# calculates the error of the candidate equation as oppossed to the data
def double_derivatives(expr, a, b, list_a, list_b, var_lists, time_double, SYSTEM):
    # function derivative values
    f_d1, f_d2 = func_derivative(expr, a, b)
    df_d1, df_d2 = calc_derivatives(f_d1, f_d2, var_lists, SYSTEM)
    d_df = indiv_derivative(df_d1, df_d2)

    #data derivaive values
    d_da, d_db = calc_time_derivatives(list_a, list_b, time_double)
    t_div = time_derivative(d_da, d_db)

    # evaluation of function values versus data, only if not arbitrary
    if sum(d_df) != 0 and sum(t_div) != 0:
        if type(d_df) == list:
            final_eval = check(t_div, d_df)
            return final_eval

    return -100


def derivatives(expr, variable_list, var_lists, basis, time_double, best_candidates, SYSTEM):
    function_evaluations = []
    penalty = 0
    # set amount of derivative parings needed for evaluation
    permutations = [(x, y) for x in variable_list for y in variable_list if x != y]
    print("for expression: ", expr)

    # if expression is to long add penalty according to Peterson
    if (len(expr.atoms()) + len(expr.atoms(Function))) > 12:
        print("Penalty added")
        penalty = 1

    # evaluate expression for each derivative pairing
    for couple in permutations:
        print("Evaluating, ", couple[0], couple[1])
        i_1 = variable_list.index(couple[0])
        i_2 = variable_list.index(couple[1])

        func_ev = double_derivatives(expr, couple[0], couple[1], var_lists[i_1], var_lists[i_2], var_lists, time_double, SYSTEM)
        function_evaluations.append(func_ev - penalty)
        print(func_ev - penalty)
        print()

        # stop if function is always going to perform worse than current best
        if (func_ev - penalty) <= best_candidates[2]:
            break

    # keep track of worst function_eval for score
    print(function_evaluations)
    func_eval = min(function_evaluations)
    return func_eval

# error function for derivative pairing as seen in SOM 2
def check(data_derivative, function_derivative):
    total = 0
    for i in range(len(data_derivative)):
        error = math.log(1 + abs(data_derivative[i] - function_derivative[i]))
        total += error
    return round(-(1/len(data_derivative))*total, 4)

# when using split function, return the side which has better results
def eval_split(dic, tree1, tree2, basis, nodes, leafs, variable_list, var_lists, time, best_candidates, SYSTEM):
    expr1, tree1 = s3.create_new_expression(dic, tree1, nodes, leafs)
    final_eval1 = derivatives(expr1, variable_list, var_lists, basis, time, best_candidates, SYSTEM)

    expr2, tree2 = s3.create_new_expression(dic, tree2, nodes, leafs)
    final_eval2 = derivatives(expr2, variable_list, var_lists, basis, time, best_candidates, SYSTEM)

    if final_eval1 > final_eval2:
        print(final_eval1, ">", final_eval2)
        tree = tree1
        expr = expr1
        final_eval = final_eval1
    else:
        print(final_eval1, "<", final_eval2)
        tree = tree2
        expr = expr2
        final_eval = final_eval2

    return expr, tree, final_eval
