from __future__ import division

import numpy as np
from mayavi import mlab
import main
import matplotlib.pyplot as plt

vertices = np.load('vertices.npy')
foot_l_mask = np.load('foot_l_mask.npy')
foot_r_mask = np.load('foot_r_mask.npy')
foot_l = np.load('foot_l.npy')
foot_r = np.load('foot_r.npy')
faces = np.load('faces.npy')

feet_mask = foot_l_mask + 2 * foot_r_mask


# CALCULATING MEDIAL AXES OF THE FEET
axis_l = np.mean(foot_l, 0)
axis_r = np.mean(foot_r, 0)

foot_ll_mask = (vertices[:, 0] <= axis_l[0]) * (vertices[:, 1] < axis_l[1]) * foot_l_mask
foot_lr_mask = (vertices[:, 0] > axis_l[0]) * (vertices[:, 1] < axis_l[1]) * foot_l_mask
foot_rl_mask = (vertices[:, 0] <= axis_r[0]) * (vertices[:, 1] < axis_r[1]) * foot_r_mask
foot_rr_mask = (vertices[:, 0] > axis_r[0]) * (vertices[:, 1] < axis_r[1]) * foot_r_mask
feet_sides_mask = foot_ll_mask + 2 * foot_lr_mask + 3 * foot_rl_mask + 4 * foot_rr_mask

# PT-1 ... LYTKO = CALF
calf_l = main.calf_point(foot_l)
calf_r = main.calf_point(foot_r)

# PT-2 ... ACHILOVKA
achill_l = main.achill_point(foot_l, foot_lr_mask, 'l')
achill_r = main.achill_point(foot_r, foot_rl_mask, 'r')

# PT-3 ... 

# mesh_vis = mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces, scalars=feet_mask.astype(np.int))
mesh_vis = mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces, scalars=feet_sides_mask.astype(np.int))
mlab.points3d(calf_l[0], calf_l[1], calf_l[2], color=(0, 0, 0), scale_factor=8)
mlab.points3d(calf_r[0], calf_r[1], calf_r[2], color=(0, 0, 0), scale_factor=8)
mlab.points3d(achill_l[0], achill_l[1], achill_l[2], color=(0, 0, 0), scale_factor=8)
mlab.points3d(achill_r[0], achill_r[1], achill_r[2], color=(0, 0, 0), scale_factor=8)
# mlab.colorbar()

mlab.show()