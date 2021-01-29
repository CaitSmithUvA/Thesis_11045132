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
# Functions which modify or create the expression tree. This is done in step 3
# of the algorithm found in the thesis
################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import operator
import math
import re
import sys


from sympy import *
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations, implicit_multiplication_application)
from sympy.parsing.mathematica import mathematica
from sympy.parsing.latex import parse_latex

# define node to build tree
class Node:
    def __init__(self, data):

        self.left = None
        self.right = None
        self.data = data

# check if s in an operator
def is_operator(s, nodes):
    if s in nodes:
        return True
    return False

# adds new node to tree, unless tree is full
def add_node(root, data, nodes):
    next_node = find_next_node(root, nodes)

    if next_node.data == None:
        # tree is full
        return True

    next_node.data = data
    return False

# finds the next free node in the tree recursively
def find_next_node(tree, nodes):
    next_node = Node(None)

    if tree.left == None:
        tree.left = Node(0)
        return tree.left

    if tree.right == None:
        tree.right = Node(0)
        return tree.right

    # traverse to left side
    if tree.left != None and is_operator(tree.left.data, nodes):
        next_node = find_next_node(tree.left, nodes)

    # if no free node has been found yet, traverse right
    if next_node.data == None and tree.right != None and is_operator(tree.right.data, nodes):
        next_node = find_next_node(tree.right, nodes)

    return next_node

# prints the expression found in the tree
def tree_display(tree):
    if tree == None or type(tree) == type(True):
        return "()"

    s = " "
    if tree.left != None:
        s += tree_display(tree.left) + " "
    s += str(tree.data) + " "
    if tree.right != None:
        s += tree_display(tree.right) + " "

    return "(" + s + ")"

# prints expression in readable manner, as well as evaluatable
def equation_display(expr, dic):
    equation = []
    for item in expr[1:-1]:
        if item in dic:
            equation.append(str(dic[item][0]))

        else:
            equation.append(str(item))

    expression_string = ''.join(equation)
    expression_string = parse_expr(str(expression_string), transformations=(standard_transformations + (implicit_multiplication_application,)))
    return expression_string

# change value of a node, keep operators operators and vice versa
def change(node, set):
    #print("change ", node.data)
    new = gen_rand_node(set)
    # ensure change
    while new == node.data:
        new = gen_rand_node(set)
    node.data = new
    #print("for ", node.data)
    return node

# mutate the tree accordig to operant/operator and position
def mutate(tree, target, nodes, leafs, index, changed = False):

    # if it should be changed, change for an operant
    leftOrRight = rnd.randrange(0, 2)
    if leftOrRight == 0:
        if is_operator(tree.left.data, nodes) == False:
            node = change(tree.left, leafs)
            tree.left = node
            changed = True
    else:
        if is_operator(tree.right.data, nodes) == False:
            node = change(tree.right, leafs)
            tree.right = node
            changed = True

    # if it should be changed, change for an operator or traverse left
    if leftOrRight == 0:
        if tree.left != None and is_operator(tree.left.data, nodes):
            index += 1
            if index == target:
                node = change(tree.left, nodes)
                tree.left = node
                changed = True
            mutate(tree.left, target, nodes, leafs, index, changed)
    else:
    # change or traverse right, unless change is already executed
        if tree.right != None and is_operator(tree.right.data, nodes):
            index += 1
            if index == target:
                node = change(tree.right, nodes)
                tree.right = node
                changed = True
            mutate(tree.right, target, nodes, leafs, index, changed)

def subtree(tree, target, nodes, basis, changed = False):

        # if it should be changed, change for an operant
        leftOrRight = rnd.randrange(0, 2)
        if leftOrRight == 0:
            if is_operator(tree.left.data, nodes) == False:
                changed = True
                new = create_tree(nodes, basis)
                tree.left = new
        else:
            if is_operator(tree.right.data, nodes) == False:
                changed = True
                new = create_tree(nodes, basis)
                tree.right = new

        # if it should be changed, change for an operator or traverse left
        if leftOrRight == 0:
            if tree.left != None and is_operator(tree.left.data, nodes) and changed==False:
                next_node = subtree(tree.left, target, nodes, basis, changed)
        else:
        # change or traverse right, unless change is already executed
            if tree.right != None and is_operator(tree.right.data, nodes) and changed==False:
                next_node = subtree(tree.right, target, nodes, basis, changed)

# merge two trees by means of an operator
def merge(tree1, tree2, nodes, operator = False):
    if operator == False:
        # use random operator for merging
        if tree1 != None and tree2 != None:
            startkey = gen_rand_node(nodes)
            expressionTree = Node(startkey)

            expressionTree.left = tree1
            expressionTree.right = tree2
        else:
            expressionTree = tree1

    # if operator given, use that
    else:
        if tree1 != None and tree2 != None:
            expressionTree = Node(operator)

            expressionTree.left = tree1
            expressionTree.right = tree2
        else:
            expressionTree = tree1

    return expressionTree

# switches two branches within tree
def switch(tree, nodes):
    if tree.left != None and tree.right != None:
        placeholder = tree.left
        tree.left = tree.right
        tree.right = placeholder
    return tree

def switch_in_tree(tree, target, nodes, index):
        next_node = Node(None)

        # if it should be changed, change for an operant
        if tree.left != None and is_operator(tree.left.data, nodes) == False:
            index += 1
            if index == target:
                switch(tree, nodes)
                return tree

        if tree.right != None and is_operator(tree.right.data, nodes) == False:
            index += 1
            if index == target:
                switch(tree, nodes)
                return tree

        # if it should be changed, change for an operator or traverse left
        if tree.left != None and is_operator(tree.left.data, nodes):
            index += 1
            if index == target:
                switch(tree, nodes)
            else:
                next_node = switch_in_tree(tree.left, target, nodes, index)

        # change or traverse right, unless change is already executed
        if next_node == None and tree.right.data != None and is_operator(tree.right.data, nodes):
            index += 1
            if index == target:
                switch(tree, nodes)
            else:
                next_node = switch_in_tree(tree.right, target, nodes, index)

        return tree

# divides tree in two from root node
def split(tree, nodes):
    if is_operator(tree.left.data, nodes) == True and is_operator(tree.right.data, nodes) == True:
        tree1 = tree.left
        tree2 = tree.right
        return tree1, tree2
    else:
        return False

# returns True if tree is valid, False for zoo or False if no division
def check_tree(tree, dic, nodes, leafs):
    if (tree == None):
        return

    if (tree.data == "/"):
        expr = tree_display(tree.right)
        expression_string = equation_display(expr, dic)
        if expression_string == 0:
            tree.right.data = gen_rand_node(leafs)

    # move deeper to the left
    check_tree(tree.left, dic, nodes, leafs)

    # move deeper to the right
    check_tree(tree.right, dic, nodes, leafs)
    return

# creates random node value out of dictionary
def gen_rand_node(dic):
    equation = []
    start = rnd.choice(list(dic.items()))
    return start[0]

#count nodes in tree
def node_count(root):
    if root == None:
        return 0

    total = 0
    if (root.left and root.right):
        total += 1

    total += (node_count(root.left) + node_count(root.right))
    return total

# creates expression tree using a random amount of input nodes until full
def create_tree(nodes, basis, random = False):
    if random == False:
        startkey = gen_rand_node(nodes)
        expressionTree = Node(startkey)
        i = 0
        full = False

        # add more nodes until the tree is full
        while full == False:
            key = gen_rand_node(basis)
            full = add_node(expressionTree, key, nodes)
            i += 1

    # if input sequence is given
    else:
        startkey = random[0]
        expressionTree = Node(startkey)

        i = 1
        full = False
        if len(random) == 1:
            full = True

        while full == False:
            if i > len(random) - 1:
                key = gen_rand_node(basis)
            else:
                key = random[i]
            full = add_node(expressionTree, key, nodes)
            i += 1
    return expressionTree

# checks if tree is valid, if it is: creates expression else mutate
def create_new_expression(dic, tree, nodes, leafs):
    check_tree(tree, dic, nodes, leafs)
    expr = tree_display(tree)
    expression_string = equation_display(expr, dic)
    return expression_string, tree
