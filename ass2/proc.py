#!/usr/bin/python
import numpy as np
from scipy.linalg import solve
import os
import cPickle as pickle
import pdb

def null(A, eps=1e-15):
    """ Computes the null space of the matrix A """
    u, s, vh = np.linalg.svd(A)
    null_space = np.compress(s <= eps, vh, axis=0)
    return null_space.T

def compute_homography(X, X_dash):
    """ Computes the homography matrix for a set of N correspondences,
        finds h such that Ah = 0
    """
    N = len(X)

    # Compute the A matrix
    A = np.zeros((2*N, 9))
    #pdb.set_trace()
    for i in xrange(0,N):
        x, y = X[i]
        x_dash, y_dash = X_dash[i]
        A[i,:] = np.array([-x, -y, -1, 0, 0, 0, x*x_dash, y*x_dash, x_dash])
        A[i+1,:] = np.array([0, 0, 0, -x, -y, -1, x*y_dash, y*y_dash, y_dash ])

    h = null(A)
    
    # Choosing first null vector
    h1 = h[:,0]
    h = np.reshape(h1, (3,3))
    return h1

if __name__ == "__main__":
    # Add a menu driven program
    print "Let's Start"
    os.system("python gui.py imgs/Dec12.jpg imgs/Jan12.jpg")
    with open('data', 'rb') as g:
        X, X_dash = pickle.load(g)
    h = compute_homography(X, X_dash)
    print h
