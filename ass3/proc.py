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
    f = null_space.T
    if f.size == 0:
        return np.array([0,0,0,0])
    return f[:,0]

def cross_product(a, b):
    """ Computes the cross product of 2 points/lines"""
    a1, a2, a3 = a
    b1, b2, b3 = b
    a1, a2, a3 = a1*1.0/a3, a2*1.0/a3, 1.0
    b1, b2, b3 = b1*1.0/b3, b2*1.0/b3, 1.0
    res = np.array([a2*b3-a3*b2, a3*b1-a1*b3, a1*b2-a2*b1])
    res /= res[2]
    return res

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
    cnt = 0
    for m,n in matches:
        if m.distance < 0.8*n.distance and cnt < 10:
            good.append(m)
            cnt += 1
    # Draw first 10 matches.

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ])
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ])

    return src_pts, dst_pts

def compute_fundamental(X, X_dash):
    """ Computes the fundamental matrix """
    N = len(X)
    A = np.zeros((N, 9))

    # TODO: normalization!
    T, T_dash = np.zeros((3,3)), np.zeros((3,3))

    mu_x1 = np.mean(X[:, 0])
    mu_y1 = np.mean(X[:, 1])
    std_x1 = np.std(X[:, 0])
    std_y1 = np.std(X[:, 1])

    mu_x2 = np.mean(X_dash[:, 0])
    mu_y2 = np.mean(X_dash[:, 1])
    std_x2 = np.std(X_dash[:, 0])
    std_y2 = np.std(X_dash[:, 1])

    Tp = np.array([[1/std_x1, 0, -mu_x1/std_x1],
                  [0, 1/std_y1, -mu_y1/std_y1],
                  [0, 0, 1]])

    Tp_dash = np.array([[1/std_x2, 0, -mu_x2/std_x2],
                  [0, 1/std_y2, -mu_y2/std_y2],
                  [0, 0, 1]])

    for i in xrange(N):
        x, y = X[i, :]
        x_dash, y_dash = X_dash[i, :]
        x, y = (x - mu_x1)/std_x1, (y - mu_y1)/std_y1
        x_dash, y_dash = (x_dash - mu_x2)/std_x2, (y_dash - mu_y2)/std_y2
        A[i,:] = np.array([x_dash*x, x_dash*y, x_dash, y_dash*x, y_dash*y, y_dash, x_dash, y_dash, 1])

    u, s, vh = np.linalg.svd(A)
    F_loose = vh[8, :].reshape((3,3))

    U, s, V = np.linalg.svd(F_loose)

    # Enforcing the constraint
    s[2] = 0
    S = np.diag(s)
    F = np.dot(U, np.dot(S, V))
    #F = F/np.linalg.norm(F)
    F = np.dot(Tp_dash.T, np.dot(F, Tp))
    f, _ = cv2.findFundamentalMat(X, X_dash)

    print "Fundamental matrix is {}".format(f)
    return f

def compute_epipoles(F):
    """ Computes epipoles e1 and e2 """
    e = null(F)
    e_dash = null(F.T)
    return e, e_dash


def draw_epipolar_lines(X, X_dash, F):
    """ Draws the epipolar lines corresponding to the keypoints """

    N = len(X)
    # For the first image
    img1 = cv2.imread("imgs/Sunrise_Lt.jpg", cv2.IMREAD_COLOR)
    img2 = cv2.imread("imgs/Sunrise_Rt.jpg", cv2.IMREAD_COLOR)

    e, e_dash = compute_epipoles(F)
    e, e_dash = e/e[2], e_dash/e_dash[2]
    print "Epipoles calculated from null space of F are {} and {}".format(e, e_dash)

    line_selection = []
    for idx in xrange(N):
        pt = np.array(list(X[idx, :])+[1])
        line = np.dot(F, pt)
        R, C, ch = img2.shape
        a, b, c = line
        p1 = np.array([-c/a, 0], np.int32)
        p2 = np.array([0, -c/b], np.int32)
        p3 = np.array([C, (-c-a*C)/b], np.int32)
        p4 = np.array([(-c-R*b)/a, R], np.int32)
        
        selected = [p1, p2, p3, p4]
        if p1.any() < 0:
            selected.remove(p1)
        if p2.any() < 0:
            selected.remove(p2)
        if p3.any() < 0:
            selected.remove(p3)
        if p4.any() < 0:
            selected.remove(p4)

        if len(selected) < 2:
            continue

        else:
            if len(line_selection) < 2:
                line_selection.append(line)
            cv2.line(img2, tuple(selected[0]), tuple(selected[1]), (0,255,0))

    e_dash_calc = cross_product(line_selection[0], line_selection[1])
    e_dash_calc /= e_dash_calc[2]

    line_selection = []
    for idx in xrange(N):
        pt = np.array(list(X_dash[idx, :])+[1])
        line = np.dot(F.T, pt)
        R, C, ch = img1.shape
        a, b, c = line
        p1 = np.array([-c/a, 0], np.int32)
        p2 = np.array([0, -c/b], np.int32)
        p3 = np.array([C, (-c-a*C)/b], np.int32)
        p4 = np.array([(-c-R*b)/a, R], np.int32)
        
        selected = [p1, p2, p3, p4]
        if p1.any() < 0:
            selected.remove(p1)
        if p2.any() < 0:
            selected.remove(p2)
        if p3.any() < 0:
            selected.remove(p3)
        if p4.any() < 0:
            selected.remove(p4)

        if len(selected) < 2:
            continue

        else:
            if len(line_selection) < 2:
                line_selection.append(line)
            cv2.line(img1, tuple(selected[0]), tuple(selected[1]), (0,255,0))

    e_calc = cross_product(line_selection[0], line_selection[1])
    e_calc /= e_calc[2]
    print "Epipoles calculated from intersection of epipolar lines are {} and {}".format(e_calc, e_dash_calc)
    print "Differences in the calculation of epipoles:"
    print "delta e: ", np.linalg.norm(e - e_calc)
    print "delta e':", np.linalg.norm(e_dash - e_dash_calc)
    cv2.imwrite("lines_1.png", img1)
    cv2.imwrite("lines_2.png", img2)

def compute_projective_mats(F, e, e_dash):
    """ Computes projective matrices using convenient relations:
        P = [I|0], P_dash = [M | m]"""

    P = np.concatenate((np.eye(3), np.zeros((3,1))), axis=1)

    e1, e2, e3 = e_dash
    e_dash_cross = np.array([[0, -e3, e2], [e3, 0, -e1], [-e2, e1, 0]])
    M = np.dot(e_dash_cross, F)

    P_dash = np.concatenate((M, np.reshape(e_dash, (3,1))), axis=1)
    print "Projective matrices are {} and {}".formati(P, P_dash)

    return P, P_dash

def linear_triangulation(X, X_dash, P, P_dash):
    """ Maps a linear triangulation to give a list of corresponding
    scene points """
    N = len(X)
    scene_points = []
    for idx in xrange(N):
        x, y = X[idx, :]
        x_dash, y_dash = X_dash[idx, :]
        A1 = np.array(x * P[2, :] - P[0, :])
        A2 = np.array(y * P[2, :] - P[1, :])
        A3 = np.array(x_dash * P_dash[2, :] - P_dash[0, :])
        A4 = np.array(y_dash * P_dash[2, :] - P_dash[1, :])
        A = np.array([A1, A2, A3, A4])

        scene_pt = null(A)
        if np.sum(scene_pt) == 0:
            continue
        scene_pt = scene_pt / scene_pt[3]
        scene_points.append(np.reshape(scene_pt, (1,4)))
    scene_points = np.vstack(scene_points)

    return scene_points

def get_depths(X, X_dash, P, P_dash):
    """ Get the depths of the scene points """
    scene_points = linear_triangulation(X, X_dash, P, P_dash)
    c1 = np.array([0,0,0,1])
    c2 = np.append(-np.dot(np.linalg.inv(P[:3, :3]), P[:, 3]), [1])

    diff_c1 = scene_points - c1
    diff_c2 = scene_points - c2

    princ_axis_1 = np.append(P[2,:3], [0])
    princ_axis_2 = np.append(P_dash[2,:3], [0])
    # Computing the dot products
    depth_c1 = np.dot(diff_c1, princ_axis_1)
    depth_c2 = np.dot(diff_c2, princ_axis_2)

    depth_c1 = np.linalg.det(P[:3,:3]) * depth_c1
    depth_c2 = np.linalg.det(P[:3,:3]) * depth_c2

    print "Depth vectors: {}, {}".format(depth_c1, depth_c2)
    return depth_c1, depth_c2

def main():
    """ Main function """

    # Compute the SIFT matches
    X, X_dash = compute_sift_matches("imgs/Sunrise_Lt.jpg", "imgs/Sunrise_Rt.jpg")

    # Computes fundamental using DLT
    F = compute_fundamental(X, X_dash)

    # Computes Fundamental using standard functions, using this for subsequent calculations
    f, _ = cv2.findFundamentalMat(X, X_dash)

    # Draw epipolar lines using the keypoints and F
    draw_epipolar_lines(X, X_dash, f)

    # Compute the epipoles
    e, e_dash = compute_epipoles(f)

    # Compute the projective matrices
    P, P_dash = compute_projective_mats(f, e, e_dash)

    # Finally, compute the depths
    depth_c1, depth_c2 = get_depths(X, X_dash, P, P_dash)

if __name__ == '__main__':
    main()
