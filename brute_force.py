################################################################################
# brute_force.py
# 29/01/2021
# Caitlin Smith
# 11045132
#
# Universiteit van Amsterdam
# Bachelor Scriptie: Approximating the Hamiltonian of the double pendulum using
# an evolutionary algorithm
#
# A replication of the Schmidt and Lipson 2009 algorithm, helper functions found
# in brute_force_functions; used for mutation of expression trees.
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

import brute_force_functions as bff
import step4_5 as s4
import step3 as s3


def main(init, dic, nodes, leafs, time_double, hours, candidates, best_candidates, variable_list, var_lists, basis, SYSTEM, exit):
    # set timer to take input time in hours
    start = time.time()
    PERIOD_OF_TIME = hours * 60 * 60

    while True:
        print("Start new trial")
        final = 0
        error_track = 0
        exit_agreement = 0
        updated = 0
        updated_list = []

        # chosen for colour mapping of algorithm
        rl = ["red", "green", "yellow", "blue", "grey", "pink", "white", "orange"]

        # while exit criteria have not been met, keep mutating and mergeing
        while exit_agreement == 0:
            for i in range(init):
                start_val = candidates[i][2]

                # mutate each expression init times before moveing on to the next
                for k in range(init):

                    # print for legibility
                    print("\n", final)
                    final += 1
                    print("\n\nPreviously: ", candidates[i][0], "\n", candidates[i][2])

                    # error handling for division by zero and stranding
                    if best_candidates[i][2] != None:
                        if candidates[i][0] == sympify("1")or candidates[i][0] == sympify("0"):
                            print("Changed due to 1 or 0")
                            candidates[i] = bff.create_candidate(variable_list, var_lists, basis, nodes, leafs, time_double, SYSTEM)
                            print(candidates[i][0])


                        if candidates[i][0].has(oo, -oo, zoo, nan):
                            print("Changed due to zoo")
                            error_track += 1
                            candidates[i] = bff.create_candidate(variable_list, var_lists, dic, nodes, leafs, time_double, SYSTEM)
                            print(candidates[i][0])

                    # set action and define temporary variables
                    action = rnd.choice(rl)
                    expr, tree, final_eval = candidates[i]

                    # mutates single node or leaf in tree
                    if action == "red" or action == "pink":
                        print("\n Mutate")
                        curr_eval = final_eval
                        curr_tree = tree
                        expr, tree, final_eval = bff.mut(curr_tree, nodes, leafs, dic, variable_list, var_lists, time_double, i, best_candidates, SYSTEM)

                    # mutate by means of merging
                    if action == "blue":
                        print("\n Merge")
                        expr, tree, final_eval = bff.mer(tree, best_candidates, nodes, leafs, dic, variable_list, var_lists, time_double, i, SYSTEM)

                    # split tree at top of tree
                    if action == "yellow" or action == "white" or action == "orange":
                        print("\n Split")
                        if s3.split(tree, nodes) != False:
                            tree1, tree2 = s3.split(tree, nodes)
                            expr, tree, final_eval = s4.eval_split(dic, tree1, tree2, dic, nodes, leafs, variable_list, var_lists, time_double, best_candidates[i], SYSTEM)

                    # switch two branches under random node
                    if action == "grey":
                        print("\n Switch")
                        expr, tree, final_eval = bff.switch(tree, expr, nodes, leafs, dic, variable_list, var_lists, time_double, i, best_candidates, SYSTEM)

                    # replaces leaf with another tree
                    if action == "green":
                        print("\n Add subtree")
                        expr, tree, final_eval = bff.add_subtree(tree, nodes, leafs, dic, variable_list, var_lists, time_double, i, best_candidates, SYSTEM)

                    # reverts back to best equation for expression i
                    if action == "black":
                        print("\n Revert to best")
                        expr, tree, final_eval = best_candidates[i]

                    # update after mutation if enhanced
                    if final_eval >= best_candidates[i][2]:
                        if final_eval > best_candidates[i][2]:
                            print("Updated")
                            updated += 1
                            updated_list.append([final_eval, best_candidates[i][2]])
                            best_candidates[i] = expr, tree, final_eval
                        flag = False
                        for j in range(len(best_candidates)):
                            if expr == best_candidates[j][0]:
                                flag = True
                        if flag == False:
                            best_candidates[i] = expr, tree, final_eval

                    # if expression too long; split, but for a maximum of five times
                    size = s3.node_count(tree)
                    tries = 0
                    while size > 18:
                        tries += 1
                        if s3.split(tree, nodes) != False:
                            tree1, tree2 = s3.split(tree, nodes)
                            expr, tree, final_eval = s4.eval_split(dic, tree1, tree2, dic, nodes, leafs, variable_list, var_lists, time_double, best_candidates[i], SYSTEM)
                        if tries >= 5:
                            expr, tree, final_eval = bff.create_candidate(variable_list, var_lists, dic, nodes, leafs, time_double, SYSTEM)
                        size = s3.node_count(tree)

                        if final_eval >= best_candidates[i][2]:
                            if final_eval > best_candidates[i][2]:
                                print("Updated")
                                updated += 1
                                updated_list.append([final_eval, best_candidates[i][2]])
                                best_candidates[i] = expr, tree, final_eval
                            flag = False
                            for j in range(len(best_candidates)):
                                if expr == best_candidates[j][0]:
                                    flag = True
                            if flag == False:
                                best_candidates[i] = expr, tree, final_eval

                    # always update candidates to ensure growth
                    # stagnates when converges on best, if not using merge
                    candidates[i] = expr, tree, final_eval

                # if after ten mutations no update, change entire expression
                if candidates[i][2] == start_val:
                    print("Created new candidate")
                    expr, tree, final_eval = bff.create_candidate(variable_list, var_lists, dic, nodes, leafs, time_double, SYSTEM)
                    if final_eval >= best_candidates[i][2]:
                        if final_eval > best_candidates[i][2]:
                            updated += 1
                            updated_list.append([final_eval, best_candidates[i][2]])
                        best_candidates[i] = expr, tree, final_eval
                    candidates[i] = expr, tree, final_eval

                if final % 100 == 0:

                    # alternate between adding reversion fucntion and removing it
                    rnd.shuffle(rl)
                    if "black" not in rl:
                        rl.append("black")
                    else:
                        rl.remove("black")

                    # show intermediate results
                    bff.print_inter(candidates, best_candidates, error_track, updated, updated_list)
                    updated = 0
                    updated_list = []

                # terminate is worst evaluation meets exit criteria
                if bff.terminate(final_eval, exit) == 1:
                    exit_agreement = 1
                    bff.pretty_print(best_candidates)
                    return best_candidates
                    break

                # show best performing function at the end of each cycle
                print(i, "best error :", best_candidates[i][2], "best expression: ", best_candidates[i][0])
                print("######################################################\n \n")

                # check time constraints
                if time.time() > start + PERIOD_OF_TIME :
                    bff.pretty_print(best_candidates)
                    return best_candidates

        return best_candidates
    return best_candidates
