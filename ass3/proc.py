#!/usr/bin/python

import cv2
import os
import numpy as np
import pdb
import cPickle as pickle

def null(A, eps=1e-7):
    """ Computes the null space of the matrix A """
    u, s, vh = np.linalg.svd(A)
    null_space = np.compress(s <= eps, vh, axis=0)
    f = vh[:,-1]
    F_loose = np.reshape(f, (3,3))

    U, s, V = np.linalg.svd(F_loose)

    # Enforcing the constraint
    s[2] = 0
    S = np.diag(s)
    F = np.dot(U, np.dot(S, V))
    return F

def compute_fundamental(X, X_dash):
    """ Computes the fundamental matrix """
    N = len(X)
    A = np.array(N, 9)
    for i in xrange(N):
        x, y = X[i]
        x_dash, y_dash = X_dash[i]
        A[i,:] = np.array([x_dash*x, x_dash*y, x_dash,
            y_dash*x, y_dash*y, y_dash, x_dash, y_dash, 1])

    f = null(A)
    return f



def main():
    """ Main function """
    os.system("python gui.py imgs/Sunrise_Lt.jpg imgs/Sunrise_Rt.jpg")
    with open("data.pk", 'rb') as g:
        X, X_dash = pickle.load(g)
    F = compute_fundamental(X, X_dash)

if __name__ == '__main__':
    main()
