################################################################################
# main.py
# 29/01/2021
# Caitlin Smith
# 11045132
#
# Universiteit van Amsterdam
# Bachelor Scriptie: Approximating the Hamiltonian of the double pendulum using
# an evolutionary algorithm
#
# main() runs the algorithm for the double pendulum approximation calling on
# polyfit.py and brute_force.py
################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import operator
import math
import copy
import time
import regex
import re

from sympy import *
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations, implicit_multiplication_application)
from sympy.parsing.mathematica import mathematica
from sympy.parsing.latex import parse_latex
from sympy import sympify

import brute_force_functions as bff
import polyfit as pf
import brute_force as bf
import step3 as s3
import S_NN_train as nnt
import visualize as v

vis = input("Run visualization? [y/n]: ")
while vis not in ["y", "n"]:
    vis = input("Run visualization? [y/n]: ")
if vis == "y":
    v.run_vis()

system_choice = input(" 1 for the single pendulum positional data \n 2 for the single pendulum angular data \n 3 for the double pendulum \n run for corresponding system: ")
while system_choice not in [str(1), str(2), str(3)]:
    system_choice = input(" 1 for the single pendulum positional data \n 2 for the single pendulum angular data \n 3 for the double pendulum \n run for corresponding system: ")


if system_choice == "1":
    SYSTEM = "SP_XY"
    exit = 0.5
elif system_choice == "2":
    SYSTEM = "SP_AB"
    exit = 0.8
else:
    SYSTEM = "DP"
    exit = 2.0

init = 10 #amount of candidate equations to be used
bf_run_time = 60 #amount of hours run time allowed

if SYSTEM == "DP":
    time_data = []
    theta_1 = []
    omega_1 = []
    theta_2 = []
    omega_2 = []

    file = open("dp_data.txt", "r")
    content = file.read().split("\n")
    for line in content[1:-1]:
        line = line.split()
        if float(line[0]) != 0:
            time_data.append(float(line[0]))
            theta_1.append(float(line[5]))
            theta_2.append(float(line[7]))
            omega_1.append(float(line[6]))
            omega_2.append(float(line[8]))

    # user input; variables to take into account
    input_var = {
        "A": ["a", 5, theta_1], # theta 1
        "B": ["b", 5, omega_1], # omega 1
        "E": ["e", 5, theta_2], # theta 2
        "F": ["f", 5, omega_2], # omega 2
        "C": ["cos(a)", 1],
        "S": ["sin(a)", 1],
    #    "I": ["cos(b)", 0],
    #    "J": ["sin(b)", 0],
        "M": ["cos(e)", 0],
        "N": ["sin(e)", 0],
    #    "U": ["cos(f)", 0],
    #    "V": ["sin(f)", 0],
        "X": ["cos(a + e)", 0]
    }

if "SP" in SYSTEM:
    time_data = []
    theta = []
    omega = []
    alpha = []
    x = []
    y = []

    file = open("generated_data.txt", "r")
    content = file.read().split("\n")
    for line in content[1:-1]:
        line = line.split()
        if float(line[0]) != 0:
            time_data.append(float(line[0]))
            x.append(float(line[1]))
            y.append(float(line[2]))
            alpha.append(float(line[3]))
            theta.append(float(line[4]))
            omega.append(float(line[5]))

    if "XY" in SYSTEM:
        input_var = {
                "X": ["x", 5, x],
                "Y": ["y", 5, y],
            #    "C": ["cos(x)", 1],
            #    "S": ["sin(x)", 1],
            #    "I": ["cos(y)", 0],
            #    "J": ["sin(y)", 0],
            }

    else:
        input_var = {
                "A": ["a", 5, theta], # theta
                "B": ["b", 5, omega], # omega
                "C": ["cos(a)", 1],
                "S": ["sin(a)", 1],
                "I": ["cos(b)", 0],
                "J": ["sin(b)", 0],
            }

basis = {
    "+": ["+", 2],
    "-": ["-", 2],
    "*": ["*", 2],
#    "/": ["/", 2],
    "1": [1, 0],
    "2": [2, 0],
    "3": [3, 0],
    "4": [4, 0],
    "5": [5, 0],
    "P": [math.pi, 0],
    "Z": [math.e, 0],
}

nodes = {
    "+": ["+", 2],
    "-": ["-", 2],
    "*": ["*", 2],
#    "**": ["**", -1],
#    "/": ["/", 2],
}

leafs = {
    "1": [1, 0],
    "2": [2, 0],
    "3": [3, 0],
    "4": [4, 0],
    "5": [5, 0],
#    "6": [6, 0],
#    "7": [7, 0],
#    "8": [8, 0],
#    "9": [9, 0],
#    "10": [10, 0],
    "P": [math.pi, 0],
#    "Z": [math.e, 0],
}

variable_list = [] # contains variables
var_lists = [] # contains data lists
candidates = []
dic, leafs, nodes = bff.create_dic(input_var, basis, nodes, leafs)

for node in dic:
    if dic[node][1] == 5:
        variable_list.append(dic[node][0])
        var_lists.append(dic[node][2])

# _______________________________POLYFIT IMPLEMENTATION_________________________
def single_pend_xy(eqn):
    expr = eqn.replace("x0", "x")
    expr = parse_expr(expr.replace("x1", "y"), evaluate=False)
    return expr

def single_pend_ab(eqn):
    expr = eqn.replace("x0", "a")
    expr = parse_expr(expr.replace("x1", "b"), evaluate=False)
    return expr

def double_pend(eqn):
    expr = eqn.replace("x0", "a")
    expr = expr.replace("x1", "e")
    expr = expr.replace("x2", "b")
    expr = parse_expr(expr.replace("x3", "f"), evaluate=False)
    return expr

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

for i in range(1, 4):
    if SYSTEM == "SP_XY":
        polyfit_result = pf.polyfit(2, "pendulum_h_1 copy.txt", 4)
        eqn = str(single_pend_xy(str(polyfit_result[0])))

    if SYSTEM == "SP_AB":
        polyfit_result = pf.polyfit(2, "pendulum_h_1 copy.txt", 4)
        eqn = str(single_pend_ab(str(polyfit_result[0])))

    if SYSTEM == "DP":
        #polyfit_result = pf.polyfit(i, "real_double_pend_h_1 copy.txt", 7) # for double pendulum provided data
        polyfit_result = pf.polyfit(i, "dp_data.txt", 7) # for double pendulum generated data
        eqn = str(double_pend(str(polyfit_result[0])))

    print(eqn)

    #translate expression from AI Feynman structure to expression tree
    eqn = eqn.replace("+", " + ")
    eqn = eqn.replace("**", " ^ ")
    eqn = eqn.replace("*", " * ")
    eqn = eqn.replace("(", "")
    eqn = eqn.replace(")", "")

    eqn_list = eqn.split("+")
    tree_lists = []
    for expr in eqn_list:
        ops = []
        variables = []
        flag = False
        values = expr.split(" ")
            #print(values)
        for i in range(len(values)):
            if flag == True:
                variables.append(values[i - 2])
                flag = False
            else:
                if values[i] in ["+", "-", "*", "/"]:
                    ops.append(values[i])
                if values[i] == "^":
                    ops.append("*")
                    flag = True
                if isfloat(values[i]) or values[i] in ["a", "b", "e", "f"]:
                    variables.append(values[i])
        tree_input = ops + variables
        tree = s3.create_tree(nodes, dic, random = tree_input)
        tree_lists.append(tree)

    while len(tree_lists) > 1:
        tree = s3.merge(tree_lists[0], tree_lists[1], nodes, "+")
        tree_lists.pop(0)
        tree_lists.pop(0)
        tree_lists.insert(0, tree)

    # create first candidate equations using the polynomails
    expr = s3.create_new_expression(dic, tree, nodes, leafs)
    final_eval = float(polyfit_result[1]) * -1
    candidates.append((expr[0], tree, final_eval))

# _______________________________BRUTE FORCE____________________________________

for can in candidates:
    if can[1] == None:
        candidates.remove(can)

# create further expressions as candidates
new_init = init - len(candidates)
new_candidates = bff.create_candidates(new_init, variable_list, var_lists, dic, nodes, leafs, time_data, SYSTEM)
candidates = candidates + new_candidates
best_candidates = [(None, None, -200)]*init

best_eq = bf.main(init, dic, nodes, leafs, time_data, bf_run_time, candidates, best_candidates, variable_list, var_lists, basis, SYSTEM, exit)

# __________________________________ANN_________________________________________
#   This would be the positioning of the ANN in the AI Feynman structure
#   which has been saved for further research
#
