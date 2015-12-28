from __future__ import division

import os

import numpy as np

def get_names(dir, ext='.ply'):
    files = os.listdir(dir)
    names = [x[:-4] for x in files if ext in x]
    return names

def get_heel_cut_plane(pts, heel_cut, max_h):
    min_x, _, _ = pts.min(0)
    max_x, _, _ = pts.max(0)
    plane_v = np.array([[min_x, heel_cut, max_h],
                        [max_x, heel_cut, max_h],
                        [max_x, heel_cut, 0],
                        [min_x, heel_cut, 0]])
    plane_f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int)

    return plane_v, plane_f
    # mlab.triangular_mesh(plane_v[:, 0], plane_v[:, 1], plane_v[:, 2], plane_f, color=(0, 1, 1), opacity=0.8)
    # mlab.triangular_mesh(plane_v[:, 0], plane_v[:, 1], plane_v[:, 2], plane_f, color=(0, 1, 1), opacity=0.8)


if __name__ == '__main__':
    dir = '/home/tomas/Data/Paty/zari/ply/'
    ext = '.ply'

    names = get_names(dir)
    print names