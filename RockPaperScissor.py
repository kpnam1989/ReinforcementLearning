# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 10:20:33 2018
# Thanks to:
# https://medium.com/@adam5ny/linear-programming-in-python-cvxopt-and-game-theory-8626a143d428
#
@author: nkieu
"""
import numpy as np 
from cvxopt import matrix, solvers

def maxmin(A, solver="glpk"):
    num_vars = len(A)
    
    # minimize c * x
    # c has 1 more elements than A
    # the first element is V
    # Notes: we include V as a variable to run the minimization over
    c = [-1] + [0 for i in range(num_vars)]
    c = np.array(c, dtype="float")
    c = matrix(c)
    
    # constraints G*x <= h
    G = np.matrix(A, dtype="float").T # reformat each variable is in a row
    G *= -1 # minimization constraint
    G = np.vstack([G, np.eye(num_vars) * -1]) # > 0 constraint for all vars
    new_col = [1 for i in range(num_vars)] + [0 for i in range(num_vars)]
    G = np.insert(G, 0, new_col, axis=1) # insert utility column
    G = matrix(G)
    
    # constraint for payoff
    # constrait for probability >> 0
    h = ([0 for i in range(num_vars)] + 
         [0 for i in range(num_vars)])
    h = np.array(h, dtype="float")
    h = matrix(h)
    
    # contraints Ax = b
    A = [0] + [1 for i in range(num_vars)]
    A = np.matrix(A, dtype="float")
    A = matrix(A)
    
    # Note that b is scalar
    b = np.matrix(1, dtype="float")
    b = matrix(b)
    
    sol = solvers.lp(c=c, G=G, h=h, A=A, b=b, solver=solver)
    return sol

A = [[0.0, 1.02, -2.66], [-1.02, 0.0, 1.03], [2.66, -1.03, 0.0]]
sol = maxmin(A=A)
print(sol["x"][1:])
