#!/bin/bash
import cv2
import pdb
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as plt

def color_correct(img1, fname):
    """ Corrects the images using gray world
    and white world approximations """
    mean_b, mean_g, mean_r = np.mean(img1[:,:,0]), np.mean(img1[:,:,1]), np.mean(img1[:,:,2])
    gray_b = img1[:,:,0] * (mean_g*1.0/mean_b)
    gray_r = img1[:,:,2] * (mean_g*1.0/mean_r)
    gray_g = img1[:,:,1]

    gray_corrected_img1 = np.concatenate((gray_b[..., np.newaxis],
        gray_g[..., np.newaxis],gray_r[..., np.newaxis]), axis=2)

    gray_corrected_img1 = np.array(gray_corrected_img1, dtype=np.uint8)

    max_b, max_g, max_r = np.max(img1[:,:,0]), np.max(img1[:,:,1]), np.max(img1[:,:,2])
    white_b = img1[:,:,0] * (max_g*1.0/max_b)
    white_r = img1[:,:,2] * (max_g*1.0/max_r)
    white_g = img1[:,:,1]

    white_corrected_img1 = np.concatenate((white_b[..., np.newaxis],
        white_g[..., np.newaxis],white_r[..., np.newaxis]), axis=2)
    white_corrected_img1 = np.array(white_corrected_img1, dtype=np.uint8)

    cv2.imwrite("out/gray_correct{}.png".format(fname), gray_corrected_img1)
    cv2.imwrite("out/white_correct{}.png".format(fname), white_corrected_img1)
    return white_corrected_img1, gray_corrected_img1

def line(p1, p2):
    """ Gives line representation between two points """
    m = (p2[1]-p1[1])*1.0/(p2[0]-p1[0])
    line = np.array([m, -1, p1[1]-m*p1[0]])
    return line*1.0/line[2]

def intersection(l1, l2):
    """ Gives line intersection """
    a, b, c = l1
    j, k, l = l2

    x = (c*k-b*l)*1.0/(b*j-a*k)
    y = (a*l-c*j)*1.0/(b*j-a*k)

    return [x, y]

def inTriangle(a1, a2, a3, p):
    """ Checks if point p lies inside triangle
    formed by a1, a2, a3 """
    p0x, p0y = a1
    p1x, p1y = a2
    p2x, p2y = a3
    px, py = p
    # Using Barycentric conventions
    area = 0.5 *(-p1y*p2x + p0y*(-p1x + p2x) + p0x*(p1y - p2y) + p1x*p2y);
    s = 1/(2*area)*(p0y*p2x - p0x*p2y + (p2y - p0y)*px + (p0x - p2x)*py)
    t = 1/(2*area)*(p0x*p1y - p0y*p1x + (p0y - p1y)*px + (p1x - p0x)*py)

    if (s > 0 and t >0 and 1-s-t > 0):
        return True
    else:
        return False

def max_sat(x, y, c1, c2, c3, k):
    """ Maximally Saturates the image """
    C = np.array([1.0/3, 1.0/3])

    L1 = line(c2, c3)
    L2 = line(c1, c3)
    L3 = line(c1, c2)
    sat = np.zeros((x.shape[0], x.shape[1], 3))

    for row in xrange(x.shape[0]):
        for col in xrange(x.shape[1]):
            pt = [x[row, col], y[row, col]]
            l = line(C, pt)
            if inTriangle(C, c1, c2, pt):
                # Opposite to c3
                sat_pt = intersection(l, L3)
            elif inTriangle(C, c2, c3, pt):
                # Opposite to c1
                sat_pt = intersection(l, L1)
            elif inTriangle(C, c3, c1, pt):
                # Opposite to c2
                sat_pt = intersection(l, L2)
            if k == 100:
                sat[row, col] = np.array(sat_pt+[1-sum(sat_pt)])
            else:
                mp = section_formula(C, sat_pt, k)
                sat[row, col] = np.array(mp+[1-sum(mp)])

            #print sat[row, col], pt
    return sat

def xyz2rgb(img_norm, sum_vec):
    """ Converts image from XYZ to RGB"""
    img = img_norm * (sum_vec)
    Tp = np.array([
        [0.0583, -0.1185, 0.8986],
        [-0.9843, 1.9984, -0.0283],
        [1.9107, -0.5326, -0.2883],
        ])
    rgb = np.dot(img, Tp.T)
    #bl = rgb[:,:,2].copy()
    #red = rgb[:,:,0].copy()
    #rgb[:,:,2] = red
    #rgb[:,:,0] = bl

    return rgb

def section_formula(pt1, pt2, k):
    """ Divides the line between PT1, PT2 in the ratio of
    k:1 """
    x1, y1 = pt1
    x2, y2 = pt2
    x3 = (k*x2+x1)*1.0/(k+1)
    y3 = (k*y2+y1)*1.0/(k+1)
    return [x3, y3]

def analyse_sat(img, fname, argum):
    """ Analyse the images in saturation domain """
    x, y, z = img[:,:,0], img[:,:,1], img[:,:,2]
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    z = z.astype(np.float64)
    nrows = img.shape[0]
    ncols = img.shape[1]
    Tp = np.array([[0.2001, 0.1736, 0.6067], [0.1143, 0.5868, 0.2988],
        [1.1149, 0.0661, 0]])
    xyz = np.dot(img, Tp.T)
    sum_vec = xyz.sum(axis=2, keepdims=True)
    xyz = xyz/sum_vec
    x = xyz[:,:,0].ravel()
    y = xyz[:,:,1].ravel()
    x[np.isnan(x)] = 0
    y[np.isnan(y)] = 0

    # Use the chromaticity model
    plt.figure()
    plt.scatter(x, y)
    plt.savefig("out/gamut_triangle{}.png".format(fname))

    a = np.min(x[np.nonzero(x)]), np.min(y[np.nonzero(y)])
    b = x.max(), y[x.argmax()]
    c = x[y.argmax()], y.max()
    print a, b, c

    img_norm = xyz * (sum_vec)
    Tp = np.array([
        [0.0583, -0.1185, 0.8986],
        [-0.9843, 1.9984, -0.0283],
        [1.9107, -0.5326, -0.2883],
        ])
    rgb = np.dot(img_norm, Tp.T)
    cv2.imwrite('out/self_recons_{}.png'.format(fname), rgb)
    if argum == "max":
        # Saturation Max
        for k in [0, 0.1, 0.3, 0.5, 0.7, 1, 2, 100]:
            sat = max_sat(xyz[:,:,0], xyz[:,:,1], a, b, c, k)
            sat_xyz = np.array(sat)
            x_sat = sat_xyz[:,:,0].ravel()
            y_sat = sat_xyz[:,:,1].ravel()
            plt.figure()
            plt.scatter(x_sat, y_sat)
            plt.savefig('out/gamut_triangle_maxsat_{}_{}.png'.format(fname, k))
            res = xyz2rgb(sat, sum_vec)
            res = np.array(res, dtype=np.uint8)
            cv2.imwrite("out/recons_{}_{}.png".format(k, fname), res)

    else:
        # Saturation Min
        print "False"

def mean_shift_seg(img):
    """ Segments the image using Mean Shift Segmentation Algorithm """


if __name__ == "__main__":
    img1 = cv2.imread("imgs/Indoor_artifical_illumination.jpg", cv2.IMREAD_COLOR)
    img2 = cv2.imread("imgs/Indoor_normal_illumination.jpg", cv2.IMREAD_COLOR)
    img3 = cv2.imread("imgs/TheraWallPainting.jpg", cv2.IMREAD_COLOR)
    res1, res2 = color_correct(img1, "artificial")
    res3, res4 = color_correct(img2, "normal")
    res5, res6 = color_correct(img3, "thera")
    analyse_sat(res1, "artificial", "max")
    analyse_sat(res3, "normal", "max")
    analyse_sat(res5, "thera", "max")
