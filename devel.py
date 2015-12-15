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

# plt.figure()
# plt.plot(foot_r[:, 0], foot_r[:, 2], 'rx')
# plt.show()

min_h = 20
max_h = 100

# PT-1 ... LYTKO = CALF
calf_l = main.calf_point(foot_l)
calf_r = main.calf_point(foot_r)

# PT-2 ... ACHILOVKA
achill_l = main.achill_point(foot_l, 'l')
achill_r = main.achill_point(foot_r, 'r')

# PT-3 & 4 ... HEEL POINTS
heel_cut_l, pts_l, mean_pt_l = main.cut_heel(foot_l, 'l')
closest_pt_l, widest_pt_l = main.heel_points(foot_l, heel_cut_l, max_h, show=False)
heel_cut_r, pts_r, mean_pt_r = main.cut_heel(foot_r, 'r')
closest_pt_r, widest_pt_r = main.heel_points(foot_r, heel_cut_r, max_h, show=False)

mesh_vis = mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces, scalars=feet_mask.astype(np.int))
# mesh_vis = mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces, scalars=feet_sides_mask.astype(np.int))

mlab.points3d(calf_l[0], calf_l[1], calf_l[2], color=(1, 1, 1), scale_factor=8)
mlab.points3d(calf_r[0], calf_r[1], calf_r[2], color=(1, 1, 1), scale_factor=8)

mlab.points3d(achill_l[0], achill_l[1], achill_l[2], color=(0, 0, 0), scale_factor=8)
mlab.points3d(achill_r[0], achill_r[1], achill_r[2], color=(0, 0, 0), scale_factor=8)

mlab.points3d(closest_pt_l[0], closest_pt_l[1], closest_pt_l[2], color=(0, 1, 1), scale_factor=8)
mlab.points3d(closest_pt_r[0], closest_pt_r[1], closest_pt_r[2], color=(0, 1, 1), scale_factor=8)

mlab.points3d(widest_pt_l[0], widest_pt_l[1], widest_pt_l[2], color=(1, 0, 1), scale_factor=8)
mlab.points3d(widest_pt_r[0], widest_pt_r[1], widest_pt_r[2], color=(1, 0, 1), scale_factor=8)


# mlab.points3d(ankle[0], ankle[1], ankle[2], color=(0, 0, 0), scale_factor=8)
# mlab.points3d(cp[0], cp[1], cp[2], color=(0, 0, 0), scale_factor=8)


# for i in range(len(pts_l)):
#     mlab.points3d(pts_l[i][0], pts_l[i][1], pts_l[i][2], color=(0, 0, 0), scale_factor=2)
# mlab.points3d(mean_pt_l[0], mean_pt_l[1], mean_pt_l[2], color=(1, 0, 1), scale_factor=4)
min_x, _, _ = foot_l.min(0)
max_x, _, _ = foot_l.max(0)
plane_v = np.array([[min_x, heel_cut_l, max_h],
                    [max_x, heel_cut_l, max_h],
                    [max_x, heel_cut_l, 0],
                    [min_x, heel_cut_l, 0]])
plane_f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int)
# mlab.triangular_mesh(plane_v[:, 0], plane_v[:, 1], plane_v[:, 2], plane_f, color=(0, 1, 1), opacity=0.8)

# for i in range(len(pts_r)):
#     mlab.points3d(pts_r[i][0], pts_r[i][1], pts_r[i][2], color=(0, 0, 0), scale_factor=2)
# mlab.points3d(mean_pt_r[0], mean_pt_r[1], mean_pt_r[2], color=(1, 0, 1), scale_factor=4)
min_x, _, _ = foot_r.min(0)
max_x, _, _ = foot_r.max(0)
plane_v = np.array([[min_x, heel_cut_r, max_h],
                    [max_x, heel_cut_r, max_h],
                    [max_x, heel_cut_r, 0],
                    [min_x, heel_cut_r, 0]])
plane_f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int)
# mlab.triangular_mesh(plane_v[:, 0], plane_v[:, 1], plane_v[:, 2], plane_f, color=(0, 1, 1), opacity=0.8)

# mlab.colorbar()

mlab.show()