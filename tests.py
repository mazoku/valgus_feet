from __future__ import division

import numpy as np

import main


def splitMesh_test():
    vertices = np.array([[2,3,0],
                         [3,3,0],
                         [1,2,0],
                         [2,2,0],
                         [3,2,0],
                         [1,1,0],
                         [2,1,0],

                         [4,2,0],
                         [3,1,0],
                         [4,1,0],
                         [5,1,0]], dtype=np.int)
    faces = np.array([[0,2,3],
                      [0,1,3],
                      [1,3,4],
                      [2,3,5],
                      [3,5,6],
                      [3,4,6],

                      [7,8,9],
                      [7,9,10]], dtype=np.int)

    mask_v = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.bool)

    f_sets, v_sets = main.splitMesh(vertices, faces, mask_v=mask_v)
    print 'faces:', f_sets
    print 'vertices:', v_sets


if __name__ == '__main__':
    splitMesh_test()