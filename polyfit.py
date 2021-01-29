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
#S_polyfit.py from AIFeynman
#https://github.com/SJ001/AI-Feynman/blob/master/aifeynman/S_polyfit.py
################################################################################

import numpy as np
import os
import polyfit_utsils as pfu
import itertools
import sys
import csv
import sympy
from sympy import symbols, Add, Mul, S, simplify
from scipy.linalg import fractional_matrix_power

def mk_sympy_function(coeffs, num_covariates, deg):
    generators = [pfu.basis_vector(num_covariates+1, i) for i in range(num_covariates+1)]
    powers = map(sum, itertools.combinations_with_replacement(generators, deg))

    coeffs = np.round(coeffs,2)

    xs = (S.One,) + symbols('z0:%d'%num_covariates)
    if len(coeffs)>1:
        return Add(*[coeff * Mul(*[x**deg for x, deg in zip(xs, power)]) for power, coeff in zip(powers, coeffs)])
    else:
        return coeffs[0]

def polyfit(maxdeg, filename, col):
    n_variables = np.loadtxt(filename, dtype='str', skiprows=1, usecols=range(2, col)).shape[1]-1
    variables = np.loadtxt(filename, usecols=(2,col), skiprows=1)
    #print(variables)
    means = [np.mean(variables)]

    for j in range(2,n_variables):
        v = np.loadtxt(filename, usecols=(j,), skiprows=1)
        means = means + [np.mean(v)]
        variables = np.column_stack((variables,v))

    f_dependent = np.loadtxt(filename, usecols=(n_variables,), skiprows=1)

    if n_variables>1:
        C_1_2 = fractional_matrix_power(np.cov(variables.T),-1/2)
        x = []
        z = []
        for ii in range(len(variables[0])):
            variables[:,ii] = variables[:,ii] - np.mean(variables[:,ii])
            x = x + ["x"+str(ii)]
            z = z + ["z"+str(ii)]

        if np.isnan(C_1_2).any()==False:
            variables = np.matmul(C_1_2,variables.T).T
            res = pfu.getBest(variables,f_dependent,maxdeg)
            parameters = res[0]
            params_error = res[1]
            deg = res[2]

            x = sympy.Matrix(x)
            M = sympy.Matrix(C_1_2)
            b = sympy.Matrix(means)
            M_x = M*(x-b)

            eq = mk_sympy_function(parameters,n_variables,deg)
            symb = sympy.Matrix(z)

            for i in range(len(symb)):
                eq = eq.subs(symb[i],M_x[i])

            eq = simplify(eq)

        else:
            res = pfu.getBest(variables,f_dependent,maxdeg)
            parameters = res[0]
            params_error = res[1]
            deg = res[2]

            eq = mk_sympy_function(parameters,n_variables,deg)
            for i in range(len(x)):
                eq = eq.subs(z[i],x[i])
            eq = simplify(eq)

    else:
        res = pfu.getBest(variables,f_dependent,maxdeg)
        parameters = res[0]
        params_error = res[1]
        deg = res[2]
        eq = mk_sympy_function(parameters,n_variables,deg)
        try:
            eq = eq.subs("z0","x0")
        except:
            pass

    return (eq, params_error)
