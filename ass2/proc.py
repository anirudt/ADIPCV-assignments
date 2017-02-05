#!/usr/bin/python
import numpy as np
from scipy.linalg import solve
import os
import cPickle as pickle
import pdb
import cv2
import pdb

def null(A, eps=1e-15):
    """ Computes the null space of the matrix A """
    u, s, vh = np.linalg.svd(A)
    null_space = np.compress(s <= eps, vh, axis=0)
    return null_space.T

def solve(A):
    """ Finds the eigenvector corresponding to the smallest eigenvalue
    of the given matrix"""
    eig_vals, eig_vecs = np.linalg.eig(np.dot(A.T, A))
    return eig_vecs[:, np.argmin(eig_vals)]

def apply_homography(img, h):
    """ Applies a homography h to img """
    nrows, ncols = img.shape
    new_img = np.zeros(img.shape)
    for i in xrange(nrows):
        for j in xrange(ncols):
            x = np.array([i, j, 1])
            x_dash = np.dot(h, x)
            x_dash = x_dash / (x_dash[3])
            if np.any(x_dash < 0):
                continue
            else:
                new_img[np.round(x_dash[0]), np.round(x_dash[1])] = img[i, j]

    return new_img


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
        
        A[i*2,:] = np.array([-x, -y, -1, 0, 0, 0, x*x_dash, y*x_dash, x_dash])
        A[i*2+1,:] = np.array([0, 0, 0, -x, -y, -1, x*y_dash, y*y_dash, y_dash ])

    h = solve(A)
    
    h = np.reshape(h, (3,3))
    return h

def compute_sift_matches(f1, f2):
    """ Expects two images filenames as input, and computes the SIFT
    descriptor matches between the two images """
    img1 = cv2.imread(f1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(f2, cv2.IMREAD_COLOR)

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1, des2)

    matched_points1, matched_points2 = [], []

    for idx, match in enumerate(matches):
        idx1, idx2 = match.trainIdx, match.queryIdx
        matched_points1.append(list(kp1[idx1].pt))
        matched_points2.append(list(kp2[idx2].pt))

    h = compute_homography(matched_points1, matched_points2)




if __name__ == "__main__":
    # Add a menu driven program
    """

    print "Let's Start"
    os.system("python gui.py imgs/Dec12.jpg imgs/Jan12.jpg")
    with open('data', 'rb') as g:
        X, X_dash = pickle.load(g)
    h = compute_homography(X, X_dash)
    print h
    """
    compute_sift_matches("imgs/Dec12.jpg", "imgs/Jan7.jpg")
