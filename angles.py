from __future__ import division

import numpy as np

if __name__ == '__main__':
    name = 'augustynova'

    fname_l = name + 'left_points.npy'
    fname_r = name + 'right_points.npy'

    points_l = np.load(fname_l)
    points_r = np.load(fname_r)

