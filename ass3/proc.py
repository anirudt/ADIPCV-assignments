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

def compute_sift_matches(f1, f2):
    """ Expects two images filenames as input, and computes the SIFT
    descriptor matches between the two images """
    img1 = cv2.imread(f1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(f2, cv2.IMREAD_GRAYSCALE)

    # Initiate SIFT detector
    sift = cv2.SIFT()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L1)
    # Match descriptors.
    matches = bf.knnMatch(des1,des2, k=2)
    # Sort them in the order of their distance.
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    # Draw first 10 matches.

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    return src_pts, dst_pts

def compute_fundamental(X, X_dash):
    """ Computes the fundamental matrix """
    N = len(X)
    A = np.array((N, 9))
    for i in xrange(N):
        pdb.set_trace()
        x, y = X[i]
        x_dash, y_dash = X_dash[i]
        A[i,:] = np.array([x_dash*x, x_dash*y, x_dash,
            y_dash*x, y_dash*y, y_dash, x_dash, y_dash, 1])

    f = null(A)
    return f

def main():
    """ Main function """
    F = compute_fundamental(compute_sift_matches("imgs/Sunrise_Lt.jpg", "imgs/Sunrise_Rt.jpg"))
    pdb.set_trace()

if __name__ == '__main__':
    main()
