#!/usr/bin/python
"""
python proc.py
"""
import numpy as np
from scipy.linalg import solve
import os
import cPickle as pickle
import pdb
import cv2

def null(A, eps=1e-7):
    """ Computes the null space of the matrix A """
    u, s, vh = np.linalg.svd(A)
    null_space = np.compress(s <= eps, vh, axis=0)
    return vh[:,-1]

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
            x_dash = x_dash / (x_dash[2])
            if np.any(x_dash < 0) or (x_dash[0] >= nrows or x_dash[1] >= ncols):
                continue
            else:
                new_img[np.floor(x_dash[0])-1, np.floor(x_dash[1])-1] = img[i, j]

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
        x, y = X[i][0]
        x_dash, y_dash = X_dash[i][0]
        
        A[i*2,:] = np.array([-x, -y, -1, 0, 0, 0, x*x_dash, y*x_dash, x_dash])
        A[i*2+1,:] = np.array([0, 0, 0, -x, -y, -1, x*y_dash, y*y_dash, y_dash ])

    h = null(A)
    
    h = np.reshape(h, (3,3))
    h = h/h[2,2]
    return h

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

    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    # Use custom version
    #dst_pts, src_pts = list(dst_pts), list(src_pts)
    #M = compute_homography(dst_pts, src_pts)

    (w, h) = img1.shape
    out = cv2.warpPerspective(img2, M, (h, w))
    #out = apply_homography(img2, M)
    cv2.imwrite("out_sift.png", out)

    return out

def scene_summarise():
    files = ['imgs/Nov17.jpg', 'imgs/Dec12.jpg', 'imgs/Jan7.jpg', 'imgs/Jan12.jpg', 'imgs/Jan22.jpg']
    # TODO: Add the manual running part and save the files

    out0 = compute_sift_matches(files[4], files[0])
    out1 = compute_sift_matches(files[4], files[1])
    out2 = compute_sift_matches(files[4], files[2])
    out3 = compute_sift_matches(files[4], files[3])
    out4 = compute_sift_matches(files[4], files[4])
    #cv2.imwrite("out/homo_manual{0}.png".format(i), out)

    # Working on differences, and adding masks to avoid
    # blackened regions
    diff0 = (out1 - out0) * (out1 > 0) * (out0 > 0)
    cv2.imwrite("out/diff0.png", diff0)
    diff1 = (out2 - out1) * (out2 > 0) * (out1 > 0)
    cv2.imwrite("out/diff1.png", diff1)
    diff2 = (out3 - out2) * (out3 > 0) * (out2 > 0)
    cv2.imwrite("out/diff2.png", diff2)
    diff3 = (out4 - out3) * (out4 > 0) * (out3 > 0)
    cv2.imwrite("out/diff3.png", diff3)

    result = (diff0+diff1+diff2+diff3)
    cv2.imwrite("out/res.png", result)

def compute_vanishing_line():

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
    scene_summarise()
