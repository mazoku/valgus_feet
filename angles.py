from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

import os
import sys

import tools
import main


def angle(v1, v2):
  return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def draw_lines(vertices, poly1, poly2, pts, new=False, show=True, ax=None):
    pt1 = np.array((pts[0, 0], pts[0, 2]))
    pt2 = np.array((pts[1, 0], pts[1, 2]))
    pt3 = np.array((pts[2, 0], pts[2, 2]))
    pt4 = np.array((pts[3, 0], pts[3, 2]))

    xmin = vertices[:, 0].min()
    xmax = vertices[:, 0].max()
    x_axis = np.linspace(xmin, xmax, 2)
    y_axis1 = poly1(x_axis)
    y_axis2 = poly2(x_axis)

    inters = seg_intersect(pt1, pt2, pt3, pt4)

    if new:
        plt.figure()
        plt.plot(vertices[:, 0], vertices[:, 2], 'bx')
        ax = plt.axis()
    plt.hold(True)
    plt.plot(x_axis, y_axis1, 'r-', linewidth=4)
    plt.plot(x_axis, y_axis2, 'g-', linewidth=4)
    for i in pts:
        plt.plot(i[0], i[2], 'ko', markersize=8)
    plt.plot(inters[0], inters[1], 'yo', markersize=8)
    # plt.axis('equal')
    plt.axis(ax)
    if show:
        plt.show()
    return ax


def perp(a) :
    b = np.empty_like(a)
    b[0] = - a[1]
    b[1] = a[0]
    return b


def seg_intersect(a1, a2, b1, b2) :
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return (num / denom.astype(float)) * db + b1


def run(pts):
    # n1 = get_line_norm_eq((pts[0, 0], pts[0, 2]), (pts[1, 0], pts[1, 2]))
    # n2 = get_line_norm_eq((pts[2, 0], pts[2, 2]), (pts[3, 0], pts[3, 2]))

    deg = 1  # fitting line -> order 1
    coeff1 = np.polyfit(pts[:2, 0], pts[:2, 2], deg)
    poly1 = np.poly1d(coeff1)
    n1 = (coeff1[0], -1, coeff1[1])

    coeff2 = np.polyfit(pts[2:, 0], pts[2:, 2], deg)
    poly2 = np.poly1d(coeff2)
    n2 = (coeff2[0], -1, coeff2[1])

    theta = np.rad2deg(angle(n1[:2], n2[:2]))

    return theta, poly1, poly2


#--------------------------------------------------------------------
if __name__ == '__main__':
    month = 'zari'
    dir = '/home/tomas/Data/Paty/' + month + '/ply/'
    names = tools.get_names(dir, ext='.ply')
    # names = ['augustynova',]
    n_files = len(names)

    angles = dict()
    for (i, name) in enumerate(names):
        print '--  Processing file %i/%i - %s  --' % (i + 1, n_files, name)
        try:
            base_name = '/home/tomas/Data/Paty/' + month + '/ply/npy/' + name
            fname_l = base_name + '_left_points.npy'
            fname_r = base_name + '_right_points.npy'

            foot_l = np.load(base_name + '_foot_l.npy')
            foot_r = np.load(base_name + '_foot_r.npy')
            points_l = np.load(fname_l)
            points_r = np.load(fname_r)

            theta_L, poly1_L, poly2_L = main.angle(points_l)
            print '\tangle L: ', theta_L

            theta_R, poly1_R, poly2_R = run(points_r)
            print '\tangle R: ', theta_R

            angles[name] = {month: (theta_L, theta_R)}

            # ax = draw_lines(foot_l, poly1_L, poly2_L, points_l, new=True, show=False)
            # draw_lines(foot_r, poly1_R, poly2_R, points_r, new=True, show=True)
        except:
            print '\tSkipped: ' + str(sys.exc_info()[0])

    fname = os.path.join(dir, 'npy', 'angles_' + month + '.npy')
    print 'Saving results to', fname
    np.save(fname, angles)


    # pts = np.array([[0, 0, 0],
    #                 [1, 0, 0],
    #                 [0, 0, 0],
    #                 [1, 0, 1]])
    # theta, n1, n2 = run(pts + 1)
    # print '%.1f' % theta
    #
    # pts = np.array([[0, 0, 0],
    #                 [1, 0, 0],
    #                 [0, 0, 0],
    #                 [0, 0, 1]])
    # theta, n1, n2 = run(pts + 1)
    # print '%.1f' % theta