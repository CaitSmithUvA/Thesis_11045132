################################################################################
# brute_force_functions.py
# 29/01/2021
# Caitlin Smith
# 11045132
#
# Universiteit van Amsterdam
# Bachelor Scriptie: Approximating the Hamiltonian of the double pendulum using
# an evolutionary algorithm
#
# The performing functions of brute force. Excecutes merging, splitting, mutating
# adding of subtrees, and switches branches. Also sets up creation of expressions
################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import operator
import math
import re
import copy
import time

from sympy import *
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations, implicit_multiplication_application)
from sympy.parsing.mathematica import mathematica
from sympy.parsing.latex import parse_latex
from sympy import sympify

import step4_5 as s4
import step3 as s3

# expands dictionary to hold all input variables
# also expands nodes and leafs
def create_dic(add, basis, nodes, leafs):
    for key in add:
        basis[key] = add[key]
    if add[key][1] == 2:
        nodes[key] = add[key]
    else:
        leafs[key] = add[key]
    return basis, leafs, nodes

# creates a single candidate expression, returns expression, tree and evaluation
def create_candidate(variable_list, var_lists, basis, nodes, leafs, time_double, SYSTEM):
    # set amount of derivative parings needed for evaluation
    permutations = [(x, y) for x in variable_list for y in variable_list if x != y]

    # create new tree and expression
    function_evaluations = []

    # avoid too high complexity for double pendulum
    try:
        expressionTree = s3.create_tree(nodes, basis)
    except:
        expressionTree = False
        print("Exception Seen & Handeld")
    while expressionTree == False:
        try:
            expressionTree = s3.create_tree(nodes, basis)
        except:
            expressionTree = False
            print("Exception Seen & Handeld")

    expr, tree = s3.create_new_expression(basis, expressionTree, nodes, leafs)
    print(expr)

    # evaluate expression for each derivative pairing
    for couple in permutations:
        print("Evaluating, ", couple[0], couple[1])
        i_1 = variable_list.index(couple[0])
        i_2 = variable_list.index(couple[1])
        func_ev = s4.double_derivatives(expr, couple[0], couple[1], var_lists[i_1], var_lists[i_2], var_lists, time_double, SYSTEM)
        function_evaluations.append(func_ev)

        # print worst of these derivative pairings and append candidate to list
        func_eval = min(function_evaluations)
    return expr, tree, func_eval

# creates first x number of candidates
def create_candidates(num, variable_list, var_lists, basis, nodes, leafs, time_double, SYSTEM):
    candidates = []
    for i in range(num):

        # create new tree and expression
        expr, tree, func_eval = create_candidate(variable_list, var_lists, basis, nodes, leafs, time_double, SYSTEM)
        candidates.append((expr, tree, func_eval))
        print(expr, ":", func_eval)
        print()
    return candidates

# mutate expression by changing one node or leaf
def mut(candidate, nodes, leafs, dic, variable_list, var_lists, time_double, i, best_candidates, SYSTEM):
    length = s3.node_count(candidate) + 2
    s3.mutate(candidate, rnd.randrange(0, length), nodes, leafs, 0)
    expr, tree = s3.create_new_expression(dic, candidate, nodes, leafs)
    final_eval = s4.derivatives(expr, variable_list, var_lists, dic, time_double, best_candidates[i], SYSTEM)
    return expr, tree, final_eval

# mutate by means of merging tree and one best_candidate tree
def mer(tree, best_candidates, nodes, leafs, dic, variable_list, var_lists, time_double, i, SYSTEM):
    chosen = rnd.randrange(0, 10)

    new_tree = s3.merge(tree, best_candidates[chosen][1], nodes)
    expr, new_tree = s3.create_new_expression(dic, new_tree, nodes, leafs)

    # avoid division by zero error
    while expr.has(oo, -oo, zoo, nan):
        new_tree = s3.merge(tree, best_candidates[chosen][1], nodes)
        expr, new_tree = s3.create_new_expression(dic, tree, nodes, leafs)

    tree = new_tree
    final_eval = s4.derivatives(expr, variable_list, var_lists, dic, time_double, best_candidates[i], SYSTEM)
    return expr, tree, final_eval

# switches two branches in tree under a random node
def switch(tree, expr, nodes, leafs, dic, variable_list, var_lists, time_double, i, best_candidates, SYSTEM):
    length = s3.node_count(tree) + 2
    new_tree = s3.switch_in_tree(tree, rnd.randrange(0, length), nodes, 0)
    expr, tree = s3.create_new_expression(dic, new_tree, nodes, leafs)

    # check validity of tree
    while expr.has(oo, -oo, zoo, nan):
        length = s3.node_count(tree) + 2
        new_tree = s3.mutate(tree, rnd.randrange(0, length), nodes, leafs, 0)
        expr, tree = s3.create_new_expression(dic, new_tree, nodes, leafs)

    # if valid, evaluate score
    final_eval = s4.derivatives(expr, variable_list, var_lists, dic, time_double, best_candidates[i], SYSTEM)
    return expr, tree, final_eval

# changes leaf to node and adds tree below new node
def add_subtree(curr_tree, nodes, leafs, dic, variable_list, var_lists, time_double, i, best_candidates, SYSTEM):
    length = s3.node_count(curr_tree) + 2
    s3.subtree(curr_tree, rnd.randrange(0, length), nodes, dic, 0)
    expr, tree = s3.create_new_expression(dic, curr_tree, nodes, leafs)

    # check validity of tree
    while expr.has(oo, -oo, zoo, nan):
        length = s3.node_count(tree) + 2
        new_tree = s3.mutate(tree, rnd.randrange(0, length), nodes, leafs, 0)
        expr, tree = s3.create_new_expression(dic, new_tree, nodes, leafs)

    # if valid, evaluate score
    final_eval = s4.derivatives(expr, variable_list, var_lists, dic, time_double, best_candidates[i], SYSTEM)
    return expr, tree, final_eval

def pretty_print(candidates):
    print("\n\n\n")
    for can in candidates:
        print("Equation: ", can[0], "\nworst: ", can[2])
        print("\n")

# prints intermediate results
def print_inter(candidates, best_candidates, error_track, updated, updated_list):
    print("\n\n Candidates")
    pretty_print(candidates)
    print("\n\n")
    print("\n\n Best Candidates")
    pretty_print(best_candidates)
    print("\n\n")
    print("Amount of zoo errors encounted: ", error_track)
    print("\n\n")
    print("Amount of updates in this round: ", updated)
    print("\n\n")
    print(updated_list)
    print("\n\n")

#score chosen according to the worst score for the goal Hamiltonian
def terminate(score, exit):
    if (0 - exit) < score < exit:
        return 1
    return 0
