from __future__ import division

__author__ = 'tomas'

import warnings

import numpy as np
# from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from mayavi import mlab
# from tvtk.api import tvtk
# from mayavi.scripts import mayavi2
# from mayavi.sources.vtk_data_source import VTKDataSource
# from mayavi.modules.surface import Surface
import transformations as trans

import sys
import os
import tools
import pickle

from sklearn import decomposition as skldec
from sklearn import cluster as sklclu
from skimage import exposure as skiexp

import cv2

from scipy import signal as scisig

from stl import mesh
import pcl

import xlsxwriter


def read_ply(fname):
    print 'Reading file ...',
    f = open(fname, 'r')
    # vertices = []  # list of vertices
    # normals = []  # list of vertex normals
    # faces = []  # list of faces (triangles)

    # reding header
    line = ''
    while 'end_header' not in line:
        line = f.readline()
        words = line.split()
        if 'vertex' in words:
            n_vertices = int(words[-1])
        elif 'face' in words:
            n_faces = int(words[-1])

    vertices = np.zeros((n_vertices, 3))
    normals = np.zeros((n_vertices, 3))
    faces = np.zeros((n_faces, 4), dtype=np.int)

    for i in range(n_vertices):
        words = f.readline().split()
        vertices[i, :] = [float(x.replace(',', '.')) for x in words[0:3]]
        normals[i, :] = [float(x.replace(',', '.')) for x in words[3:]]
    for i in range(n_faces):
        words = f.readline().split()
        faces[i, :] = [int(x) for x in words]

    print 'done'
    return vertices, faces, normals


def read_stl(fname):
    print 'Reading file ...',
    model = mesh.Mesh.from_file(fname)
    print 'done'
    # class Model():
    #     def __init__(self):
    #         pass
    # model = Model()
    # model.v0 = np.array([[0, 1, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [0, 0, 0], [0, 0, 0]])
    # model.v1 = np.array([[1, 0, 1], [0, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 0], [0, 0, 1]])
    # model.v2 = np.array([[0, 0, 1], [1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 1]])
    #
    # model.vectors = np.zeros((6, 3, 3))
    # # Top of the cube
    # model.vectors[0, :, :] = np.array([[0, 1, 1], [1, 0, 1], [0, 0, 1]])
    # model.vectors[1, :, :] = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 1]])
    # # Right face
    # model.vectors[2, :, :] = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0]])
    # model.vectors[3, :, :] = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 0]])
    # # Left face
    # model.vectors[4, :, :] = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1]])
    # model.vectors[5, :, :] = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 1]])
    #
    # # model.normals = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0]])
    # model.normals = np.array([[0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, -1], [0, 0, -1]])  # faces normals
    # # model.normals = np.array([[0, -1, 0], [0, 1, 1], [0, 0, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 0, 1]])  # vertices normals

    print 'Reformatting model ...'
    print '\tvertices ...',
    vertices = np.vstack((model.v0, model.v1, model.v2))
    tmp = np.ascontiguousarray(vertices).view(np.dtype((np.void, vertices.dtype.itemsize * vertices.shape[1])))
    _, idx = np.unique(tmp, return_index=True)
    vertices = vertices[idx]
    # vertices = np.unique(vertices.view(np.dtype((np.void, vertices.dtype.itemsize*vertices.shape[1])))).view(vertices.dtype).reshape(-1, vertices.shape[1])
    print 'done'
    faces = np.zeros((model.vectors.shape[0], 3), dtype=np.int)
    print '\tfaces ...',
    for i in range(faces.shape[0]):
        for pt in range(3):
            faces[i, pt] = np.argwhere((vertices == model.vectors[i, pt, :]).sum(1) == 3)
    print 'done'
    print 'done'

    print 'Calculating normals ...',
    # computing normals
    normals_f = model.normals.copy()

    normals_v = np.zeros((vertices.shape[0], 3))
    for i in range(vertices.shape[0]):
        norms = model.normals[np.nonzero((faces == i).sum(1))[0], :]
        normals_v[i, :] = np.mean(norms, 0)
    print 'done'

    return vertices, faces, normals_v, normals_f


def alert():
    freqs = [2000, 4000, 2000]
    durs = [0.3, 0.5, 0.3]
    for (f, d) in zip(freqs, durs):
        os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % ( d, f))


def dihedral_angles(vec, plane='XY'):
    if plane == 'XZ':
        n = np.array([0, 1, 0])
    elif plane == 'YZ':
        n = np.array([1, 0, 0])
    else:
        n = np.array([0, 0, 1])

    if len(vec.shape) == 1:  # only one vector
        vec = np.expand_dims(vec, axis=0)

    dihs = np.zeros(vec.shape[0])
    for i in range(vec.shape[0]):
        try:
            dih = np.arcsin(np.dot(n, vec[i,:]) / (np.linalg.norm(n) * np.linalg.norm(vec[i,:])))
        except RuntimeWarning:
            dih = 0
        # if dih is None or dih is np.Infinity:
        #     dih = 0
        dih = np.degrees(dih)
        dih = min(dih, 180 - dih)
        dihs[i] = dih

    return dihs


def align_with_desk(vertices, idxs=None):
    if idxs is None:
        idxs = np.ones(vertices.shape[0], dtype=np.bool)
    vertices_i = vertices[idxs,...]
    cloud = pcl.PointCloud(vertices_i.astype(np.float32))
    # vertices, faces, normals_v, normals_f = read_stl(fname)

    # plane segmentation
    seg = cloud.make_segmenter()
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(5)

    # model: ax + by + cz + d = 0
    indices, model = seg.segment()
    model = [-x for x in model]

    labels = np.zeros(vertices.shape[0], dtype=np.bool)
    labels[np.nonzero(idxs)[0][np.nonzero(indices)[0]]] = 1

    u, v, w = model[:3]
    fac1 = np.sqrt(u**2 + v**2)
    Txz = np.matrix([[u / fac1, v / fac1, 0, 0],
                    [-v / fac1, u / fac1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])  # vector to xz-plane
    fac2 = np.linalg.norm(model[:3])
    Tz = np.matrix([[w / fac2, 0, -fac1 / fac2, 0],
                   [0, 1, 0, 0],
                   [fac1 / fac2, 0, w / fac2, 0],
                   [0, 0, 0, 1]])  # vector in xz-plane to z-axis
    TM = Tz * Txz

    # vertices = np.dot(vertices, TM)
    vertices = np.array(np.dot(vertices, TM[:3, :3].T))

    return vertices, labels


def plane_as_xy_transform(pts, plane, table_labs):
    # plane: ax + by - z + d = 0, norm = (a, b, -1)
    u, v, w = plane[:3]
    fac1 = np.sqrt(u**2 + v**2)
    Txz = np.matrix([[u / fac1, v / fac1, 0, 0],
                    [-v / fac1, u / fac1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])  # vector to xz-plane
    fac2 = np.linalg.norm(plane[:3])
    Tz = np.matrix([[w / fac2, 0, -fac1 / fac2, 0],
                   [0, 1, 0, 0],
                   [fac1 / fac2, 0, w / fac2, 0],
                   [0, 0, 0, 1]])  # vector in xz-plane to z-axis
    TM = Tz * Txz

    # pts = np.dot(pts, TM)
    pts = np.array(np.dot(pts, TM[:3, :3].T))

    # shift the plane to zero height
    median_z = np.median(pts[table_labs,2])
    pts[:, 2] -= median_z

    return pts


def splitMesh(vertices, faces, mask_v=None, talk=False):
    f_sets = np.zeros(faces.shape[0], dtype=np.int)
    v_sets = np.zeros(vertices.shape[0], dtype=np.int)
    current_set = 0

    # masking the faces
    if mask_v is not None:
        f_m_out_inds = np.in1d(faces, np.nonzero(mask_v == 0)[0]).reshape(faces.shape[0], 3).any(1)
        f_sets[f_m_out_inds] = -1
        v_sets[mask_v == 0] = -1

    while (f_sets == 0).any():
        next_avail_face = np.nonzero(f_sets == 0)[0][0]

        current_set += 1
        if talk:
            print 'Connecting set #%i...' % current_set,

        open_vertices = faces[next_avail_face, :]
        while open_vertices.any():
            avail_face_inds = np.nonzero(f_sets == 0)[0]
            avail_face_sub = np.in1d(faces[avail_face_inds, :], open_vertices).reshape(avail_face_inds.shape[0], 3).any(1)
            f_sets[avail_face_inds[avail_face_sub]] = current_set
            verts_inds = np.unique(faces[avail_face_inds[avail_face_sub], :].flatten())
            v_sets[tuple((verts_inds,))] = current_set
            open_vertices = faces[avail_face_inds[avail_face_sub], :]
        if talk:
            print 'done. Set #%i has %i faces.' % (current_set, (f_sets == current_set).sum())

    f_sets -= 1
    v_sets -= 1
    f_sets[f_m_out_inds] = -1
    v_sets[mask_v == 0] = -1
    return f_sets, v_sets


def pca(vertices, faces, labels, n_comps=2, show=False):
    labeled_v = vertices[labels, ...]
    vertices_n = np.zeros_like(vertices)

    if n_comps == 2:
        data = labeled_v[:, :2]
    elif n_comps == 3:
        data = labeled_v

    pca = skldec.PCA(n_components=n_comps)
    pca.fit(data)

    if n_comps == 2:
        # vertices_n[:, :2] = pca.inverse_transform(vertices[:, :2])
        vertices_n[:, :2] = pca.transform(vertices[:, :2])
        vertices_n[:, 2] = vertices[:, 2]
    else:
        # vertices_n = pca.inverse_transform(vertices)
        vertices_n = pca.transform(vertices)

    # plt.figure()
    # plt.plot(vertices_n[:, 0], vertices_n[:, 1], 'rx')
    # plt.axes().set_aspect('equal')
    # plt.show()

    if show:
        mlab.triangular_mesh(vertices_n[:, 0], vertices_n[:, 1], vertices_n[:, 2], faces, scalars=labels.astype(np.int))
        mlab.show()

    return vertices_n


def align_xy_axes(pts, labels, show=False, talk=False):
    labeled_v = pts[labels, ...]

    min_en = np.Inf
    min_en_rot = None
    ens = []
    for i in range(0, 180, 1):
        TM = trans.axangle2mat((0, 0, 1), np.deg2rad(i))
        pts_t = np.array(np.dot(labeled_v, TM.T))

        en = get_energy(pts_t, show=False)
        ens.append(en)
        if en < min_en:
            min_en = en
            min_en_rot = i

        # plt.figure()
        # plt.plot(labeled_v[:, 0], labeled_v[:, 1], 'rx')
        # plt.hold(True)
        # plt.plot(pts_t[:, 0], pts_t[:, 1], 'bx')
        # plt.axes().set_aspect('equal')
        # plt.show()

    if talk:
        print 'Min energy: %.2f, rotation angle:%i degs' % (min_en, min_en_rot)

    # points rotation
    TM = trans.axangle2mat((0, 0, 1), np.deg2rad(min_en_rot))
    pts_t = np.array(np.dot(pts, TM.T))

    if show:
        plt.figure()
        plt.subplot(221)
        plt.plot(pts[:, 0], pts[:, 1], 'rx'), plt.title('input')
        # plt.axes().set_aspect('equal')
        plt.subplot(222)
        plt.plot(pts_t[:, 0], pts_t[:, 1], 'bx'), plt.title('aligned')
        # plt.axes().set_aspect('equal')

        # plt.figure()
        # plt.plot(range(0, 180, 1), ens, 'b-', linewidth=3), plt.title('energie rotace')

    # if the feets are above each other, we have to rotate them 90 degrees
    labeled_v = pts_t[labels, ...]
    hist_x, bins_x = skiexp.histogram(labeled_v[:, 0], nbins=256)
    hist_y, bins_y = skiexp.histogram(labeled_v[:, 1], nbins=256)
    len_x = bins_x[-1] - bins_x[0]
    len_y = bins_y[-1] - bins_y[0]
    if len_x > len_y:
        cent_y = (bins_x.max() + bins_x.min()) / 2

        ang = 90
        if talk:
            print 'Rotating 90 degs (to have the feet next to each other).'
        TM = trans.axangle2mat((0, 0, 1), np.deg2rad(ang))
        pts_t = np.array(np.dot(pts_t, TM.T))
    else:
        cent_y = (bins_y.max() + bins_y.min()) / 2

    if show:
        plt.subplot(223)
        plt.plot(pts_t[:, 0], pts_t[:, 1], 'bx'), plt.title('to be next to each other')
        # plt.axes().set_aspect('equal')

    # if the feets are facing down, we have to rotate them 180 degrees
    pts_tmp = pts_t[pts_t[:, 2] > 100, :]
    # hist_y, bins_y = skiexp.histogram(pts_tmp[:, 1], nbins=256)
    mean_y = pts_tmp[:, 1].mean()

    # cent_y = (bins_y[-1] + bins_y[0]) / 2
    # if bins_y[np.argmax(hist_y)] > cent_y:
    if mean_y > cent_y:
        ang = 180
        if talk:
            print 'Rotating 180 degs (to have the feet facing up).'
        TM = trans.axangle2mat((0, 0, 1), np.deg2rad(ang))
        pts_t = np.array(np.dot(pts_t, TM.T))

    if show:
        if talk:
            print 'mean(y)=%.1f, cen=%.1f' % (bins_y[np.argmax(hist_y)], cent_y)
        plt.subplot(224)
        plt.plot(pts_t[:, 0], pts_t[:, 1], 'bx'), plt.title('to face up')
        # plt.axes().set_aspect('equal')
        plt.show()

    return pts_t


def get_energy(pts, show=False):
    hist_x, bins_x = skiexp.histogram(pts[:, 0], nbins=256)
    hist_y, bins_y = skiexp.histogram(pts[:, 1], nbins=256)

    std_x = np.std(hist_x)
    std_y = np.std(hist_y)

    len_x = bins_x[-1] - bins_x[0]
    len_y = bins_y[-1] - bins_y[0]

    # print 'std_x=%.2f, std_y=%.2f, len_x=%.2f, len_y=%.2f' % (std_x, std_y, len_x, len_y)

    # en = 10000 / std_x + 10000 / std_y + len_x + len_y
    en = len_x + len_y

    if show:
        plt.figure()
        plt.subplot(121), plt.plot(pts[:, 0], pts[:, 1], 'bx')
        plt.subplot(222), plt.plot(bins_x, hist_x, 'b-'), plt.title('X-projection')
        plt.title('std=%.2f, len=%i'%(std_x, len_x))
        plt.subplot(224), plt.plot(bins_y, hist_y, 'b-'), plt.title('Y-projection')
        plt.title('std=%.2f, len=%i'%(std_y, len_y))
        plt.show()

    return en


def planefit(vertices):
    # Fits a plane to a point cloud,
    # Where Z = aX + bY + c
    # Rearanging Eqn1: aX + bY -Z +c =0
    # Gives normal (a,b,-1)
    # Normal = (a,b,-1)
    [rows, cols] = vertices.shape
    G = np.ones((rows, 3))
    G[:, 0] = vertices[:, 0]  #X
    G[:, 1] = vertices[:, 1]  #Y
    Z = vertices[:, 2]
    (a, b, c), resid, rank, s = np.linalg.lstsq(G, Z)
    plane = (a, b, -1, c)
    # normal = (a, b, -1)
    # nn = np.linalg.norm(normal)
    # normal = normal / nn
    # return normal
    return plane


def calf_point(vertices, step=1):
    y_idxs = vertices[:, 1] < ((vertices[:, 1].max() + vertices[:, 1].min()) / 2)
    z_idxs = vertices[:, 2] > (vertices[:, 2].max() / 2)
    idxs = y_idxs * z_idxs
    vertices = vertices[idxs, :]

    start = 0
    end = int(np.ceil(vertices[:, 2].max()))
    step = 1
    widths = []

    for i in range(start, end, step):
        lower = i
        upper = i + step
        inds = (vertices[:, 2] > lower) * (vertices[:, 2] <= upper)
        pts = vertices[inds, :]
        if pts.any():
            left = (pts[:, 0]).min()
            right = (pts[:, 0]).max()
            width = right - left
        else:
            width = 0
        widths.append(width)

        # print 'bounds = (%.1f, %.1f), l = %.1f, r = %.1f, width = %.1f' % (lower, upper, left, right, width)

    widths = np.array(widths)
    # max_w = widths.max()
    max_w_ind = widths.argmax()
    calf_z = max_w_ind + step / 2

    lower = max_w_ind
    upper = max_w_ind + step
    inds = (vertices[:, 2] > lower) * (vertices[:, 2] <= upper)
    pts = vertices[inds, :]
    calf_x = (pts[:, 0].min() + pts[:, 0].max()) / 2
    calf_y = pts[:, 1].min()

    return calf_x, calf_y, calf_z


def under_ankle_point(vertices, foot_side, eps=3):
    cut, pts_band, pt_cut = under_ankle_cut(vertices, foot_side)

    inds = (vertices[:, 2] > (cut - eps)) * (vertices[:, 2] < (cut + eps))
    band = vertices[inds, :]

    closest_pt = band[band[:, 1].argmin(), :]

    return closest_pt, band


def ankle(vertices, foot_side, ankle_side, dir='ub', min_h=20, max_h=100, show=False, show_now=True):
    if foot_side in ['l', 'L', 'left']:
        is_left = True
    elif foot_side in ['r', 'R', 'right']:
        is_left = False
    else:
        raise IOError('Wrong foot side type.')
    if ankle_side in ['in', 'inner', 'i', 'I']:
        is_inner = True
    elif ankle_side in ['out', 'outer', 'o', 'O']:
        is_inner = False
    else:
        raise IOError('Wrong ankle type.')

    max_y = vertices[:, 1].max()
    min_y = vertices[:, 1].min()
    cent_y = min_y + (max_y - min_y) / 3
    vertices = vertices[vertices[:, 1] < cent_y, :]

    step_z = 2
    num_z = (max_h - min_h) / step_z
    disc_z = np.linspace(min_h, max_h, num_z)
    inds_z = np.digitize(vertices[:, 2], disc_z)

    widths = []
    for i in range(1, inds_z.max()):
        pts = vertices[inds_z == i, :]
        if pts.any():
            if (is_left and is_inner) or (not is_left and not is_inner):
                w = pts[:, 0].max()
            elif (not is_left and is_inner) or (is_left and not is_inner):
                w = pts[:, 0].min()
        else:
            if len(widths) > 0:
                w = widths[-1]
            else:
                w = 0
        widths.append(w)

    widths = np.array(widths)

    ankle_x, ankle_z, ind = detect_peak(widths, disc_z[:-1], is_left, is_inner, dir=dir)
    if (is_left and is_inner) or (not is_left and not is_inner):
        ankle_y = vertices[inds_z == ind, 1].max()
    elif (not is_left and is_inner) or (is_left and not is_inner):
        ankle_y = vertices[inds_z == ind, 1].min()

    if show:
        plt.figure()
        plt.plot(widths, disc_z[:-1], 'b-')
        plt.plot(ankle_x, ankle_z, 'mo')
        if show_now:
            plt.show()

    return ankle_x, ankle_y, ankle_z


def under_ankle_cut(vertices, foot_side, eps=3, diff_t=0.2):
    if foot_side in ['l', 'L', 'left']:
        is_left = True
    elif foot_side in ['r', 'R', 'right']:
        is_left = False
    else:
        raise IOError('Wrong foot side.')

    outers = ankle_line(vertices, foot_side, 'outer')
    mean_pt = np.median(np.array(outers), 0)

    if is_left:
        vertices = vertices[vertices[:, 0] < vertices[:, 0].mean(), :]
    else:
        vertices = vertices[vertices[:, 0] > vertices[:, 0].mean(), :]

    min_z = 20
    max_z = mean_pt[2]
    step_z = 2
    num_z = (max_z - min_z) / step_z
    disc_z = np.linspace(min_z, max_z, num_z)
    inds_z = np.digitize(vertices[:, 2], disc_z)

    pts_band = []
    for i in range(1, inds_z.max()):
        pts = vertices[inds_z == i, :]
        pts = [x for x in pts if (x[1] > mean_pt[1] - eps) and (x[1] < mean_pt[1] + eps)]
        # pts_band += pts
        # pts_band.append(np.mean(pts, 0))
        if pts:
            pts_band.append(np.array(pts).mean(0))

    # jedu odspodu a kontroluji velikost zmeny
    for i in range(len(pts_band) - 1):
        pt1 = pts_band[i]
        pt2 = pts_band[i + 1]
        diff = pt2[0] - pt1[0]
        if not is_left:
            diff *= -1
        if diff < diff_t:
            break
    cut = pts_band[i][2]
    pt_cut = pts_band[i]

    return cut, pts_band, pt_cut


def achill_point(vertices, foot_side, ankle_type, min_h=20, max_h=100, eps=0.5):
    # ankle_i = inner_ankle(vertices, side)
    # ankle_pt = ankle(vertices, foot_side, ankle_type, min_h=min_h, max_h=max_h)
    outers = ankle_line(vertices, foot_side, ankle_type)
    ankle_pt = np.median(np.array(outers), 0)

    inds = (vertices[:, 2] > (ankle_pt[2] - eps)) * (vertices[:, 2] < (ankle_pt[2] + eps))
    pts = vertices[inds, :]

    # plt.figure()
    # plt.plot(pts[:,0], pts[:,1], 'rx')
    # plt.title('%i pts'%pts.shape[0])
    # plt.show()

    min_ind = pts[:, 1].argmin()
    achill_x = pts[min_ind, 0]
    achill_y = pts[min_ind, 1]
    achill_z = ankle_pt[2]

    return achill_x, achill_y, achill_z


# def point_B(vertices, foot_side, min_h=20, max_h=100, eps=0.5):
#     ankle_o = ankle(vertices, foot_side, 'outer', min_h=min_h, max_h=max_h)
#     inds = (vertices[:, 2] > (ankle_o[2] - eps)) * (vertices[:, 2] < (ankle_o[2] + eps))
#     pts = vertices[inds, :]
#
#     # plt.figure()
#     # plt.plot(pts[:,0], pts[:,1], 'rx')
#     # plt.title('%i pts'%pts.shape[0])
#     # plt.show()
#
#     min_ind = pts[:, 1].argmin()
#     achill_x = pts[min_ind, 0]
#     achill_y = pts[min_ind, 1]
#     achill_z = ankle_o[2]
#
#     return achill_x, achill_y, achill_z


def detect_peak(data, data_ax, is_left, is_inner, win_w=4, dir='bu', show=False):
    if dir == 'ub':
        data = data[::-1]
        data_ax = data_ax[::-1]
    for i in range(len(data)):
        win_ind = [x for x in range(i + 1, i + 1 + win_w) if x < len(data)]
        if (is_left and is_inner) or (not is_left and not is_inner):
            passing = (data[win_ind] > data[i]).any()
        elif (is_left and not is_inner) or (not is_left and is_inner):
            passing = (data[win_ind] < data[i]).any()
        if not passing:
            break

    peak = (data[i], data_ax[i], i)
    # peak = (data[i], (data_ax[i] + data_ax[i + 1]) / 2, i)

    if show:
        plt.figure()
        plt.plot(data, data_ax, 'bx')
        plt.hold(True)
        plt.plot(data[i], data_ax[i], 'ro')
        plt.show()

    return peak


def ankle_line(vertices, foot_side, ankle_side):
    idxs = vertices[:, 1] < ((vertices[:, 1].max() + vertices[:, 1].min()) / 2)
    vertices = vertices[idxs, :]

    if foot_side in ['l', 'L', 'left']:
        is_left = True
    elif foot_side in ['r', 'R', 'right']:
        is_left = False
    else:
        raise IOError('Wrong foot side.')
    if ankle_side in ['in', 'inner', 'i', 'I']:
        is_inner = True
    elif ankle_side in ['out', 'outer', 'o', 'O']:
        is_inner = False
    else:
        raise IOError('Wrong ankle type.')

    min_z = 20
    max_z = 100
    step_z = 1
    num_z = (max_z - min_z) / step_z
    disc_z = np.linspace(min_z, max_z, num=num_z)

    outers = []
    inds_z = np.digitize(vertices[:, 2], disc_z)
    for i in range(1, inds_z.max() + 1):
        pts = vertices[inds_z == i, :]
        if pts.any():
            # if is_left:
            #     ind = pts[:, 0].argmin()
            # else:
            #     ind = pts[:, 0].argmax()
            if (is_left and is_inner) or (not is_left and not is_inner):
                ind = pts[:, 0].argmax()
            else:
                ind = pts[:, 0].argmin()
            outers.append(pts[ind, :])

    return outers


def cut_heel(vertices, foot_side, max_h):
    pts = ankle_line(vertices, foot_side)
    mean_pt = np.median(np.array(pts), 0)
    closest_heel = vertices[vertices[:, 2] <= max_h, 1].min()
    heel_cut = closest_heel + 0.33 * (mean_pt[1] - closest_heel)

    return heel_cut, pts, mean_pt


def heel_points(vertices, heel_cut, max_h, eps=2, show=False):
    inds = (vertices[:, 1] > (heel_cut - eps)) * (vertices[:, 1] < (heel_cut + eps)) * (vertices[:, 2] < max_h)
    heel_cnt = vertices[inds, :]

    inds_heel = (vertices[:, 1] <= heel_cut) * (vertices[:, 2] < max_h)
    heel_pts = vertices[inds_heel, :]
    closest_pt = heel_pts[heel_pts[:, 1].argmin(), :]

    min_z = heel_cnt[:, 2].min()
    max_z = heel_cnt[:, 2].max()
    step_z = 0.5
    num_z = (max_z - min_z) / step_z
    disc_z = np.linspace(min_z, max_z, num=num_z)
    inds_z = np.digitize(heel_cnt[:, 2], disc_z)

    if show:
        plt.figure()
        plt.plot(heel_cnt[:, 0], heel_cnt[:, 2], 'bx')
        plt.hold(True)

    max_w = 0
    widest_pt = None
    mean_x = heel_cnt[:, 0].mean()
    for i in range(1, inds_z.max() + 1):
        pts = heel_cnt[inds_z == i, :]
        if pts.any():
            min_x = pts[:, 0].min()
            max_x = pts[:, 0].max()
            if min_x > mean_x or max_x < mean_x:
                continue
            if show:
                plt.plot(min_x, disc_z[i - 1], 'ro')
                plt.plot(max_x, disc_z[i - 1], 'go')
            if max_x - min_x > max_w:
                max_w = max_x - min_x
                widest_pt = [(max_x + min_x) / 2, heel_pts[:, 1].min(), (disc_z[i - 1] + disc_z[i]) / 2]

    if show:
        plt.plot(widest_pt[0], widest_pt[2], 'mo')
        plt.show()

    return closest_pt, widest_pt


def widest_point(vertices, heel_cut, max_h, eps=2, show=False, show_now=True):
    inds = (vertices[:, 1] > (heel_cut - eps)) * (vertices[:, 1] < (heel_cut + eps)) * (vertices[:, 2] < max_h)
    heel_cnt = vertices[inds, :]

    inds_heel = (vertices[:, 1] <= heel_cut) * (vertices[:, 2] < max_h)
    heel_pts = vertices[inds_heel, :]

    min_z = heel_cnt[:, 2].min()
    max_z = heel_cnt[:, 2].max()
    step_z = 0.5
    num_z = (max_z - min_z) / step_z
    disc_z = np.linspace(min_z, max_z, num=num_z)
    inds_z = np.digitize(heel_cnt[:, 2], disc_z)

    if show:
        plt.figure()
        plt.plot(heel_cnt[:, 0], heel_cnt[:, 2], 'bx')
        plt.hold(True)

    max_w = 0
    widest_pt = None
    mean_x = heel_cnt[:, 0].mean()
    for i in range(1, inds_z.max() + 1):
        pts = heel_cnt[inds_z == i, :]
        if pts.any():
            min_x = pts[:, 0].min()
            max_x = pts[:, 0].max()
            if min_x > mean_x or max_x < mean_x:
                continue
            if show:
                plt.plot(min_x, disc_z[i - 1], 'ro')
                plt.plot(max_x, disc_z[i - 1], 'go')
            if max_x - min_x > max_w:
                max_w = max_x - min_x
                widest_pt = [(max_x + min_x) / 2, heel_pts[:, 1].min(), (disc_z[i - 1] + disc_z[i]) / 2]

    if show:
        plt.plot(widest_pt[0], widest_pt[2], 'mo')
        if show_now:
            plt.show()

    return widest_pt


def angle(pts):
    deg = 1  # fitting line -> order 1
    coeff1 = np.polyfit(pts[:2, 0], pts[:2, 2], deg)
    poly1 = np.poly1d(coeff1)
    n1 = (coeff1[0], -1, coeff1[1])

    coeff2 = np.polyfit(pts[2:, 0], pts[2:, 2], deg)
    poly2 = np.poly1d(coeff2)
    n2 = (coeff2[0], -1, coeff2[1])

    theta = np.arccos(np.dot(n1[:2], n2[:2]) / (np.linalg.norm(n1[:2]) * np.linalg.norm(n2[:2])))
    theta = np.rad2deg(theta)

    if theta < 90:
        theta = 180 - theta

    return theta, poly1, poly2


def perp(a) :
    b = np.empty_like(a)
    b[0] = - a[1]
    b[1] = a[0]
    return b


def line_intersect(a1, a2, b1, b2) :
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return (num / denom.astype(float)) * db + b1


def draw_lines(vertices, poly1, poly2, pts, size=(8, 11), show=False):
    pts = np.array(pts)
    pt1 = np.array((pts[0, 0], pts[0, 2]))
    pt2 = np.array((pts[1, 0], pts[1, 2]))
    pt3 = np.array((pts[2, 0], pts[2, 2]))
    pt4 = np.array((pts[3, 0], pts[3, 2]))

    xmin = vertices[:, 0].min()
    xmax = vertices[:, 0].max()
    zmin = vertices[:, 2].min()
    zmax = vertices[:, 2].max()
    x_axis = np.linspace(xmin, xmax, 2)
    y_axis1 = poly1(x_axis)
    y_axis2 = poly2(x_axis)

    inters = line_intersect(pt1, pt2, pt3, pt4)

    fig = plt.figure(figsize=size)
    # fig = plt.figure()
    plt.plot(vertices[:, 0], vertices[:, 2], 'bx')
    # ax = plt.axis()
    plt.hold(True)
    plt.plot(x_axis, y_axis1, 'r-', linewidth=2)  # line #1
    plt.plot(x_axis, y_axis2, 'g-', linewidth=2)  # line #2
    for i in pts:  # points
        plt.plot(i[0], i[2], 'ko')
    plt.plot(inters[0], inters[1], 'yo')  # intersection point
    plt.axis('equal')
    plt.axis([xmin - 10, xmax + 10, zmin - 10, zmax + 10])

    if show:
        plt.show()

    return fig


def process_filenames(all_names_dir, proc_names_dir, ext='.npy'):
    all_names = tools.get_names(all_names_dir)

    proc_files = tools.get_names(proc_names_dir, ext=ext)
    proc_names = [x[:-6] for x in proc_files if '_faces' in x]

    # failed_names = []
    # for i in all_names:
    #     if i not in proc_names:
    #         failed_names.append(i)

    failed_names = [x for x in all_names if x not in proc_names]

    return all_names, proc_names, failed_names


def get_angles(normals_v, planes=['xy', 'xz', 'yz']):
    dih_xy = []
    dih_xz = []
    dih_yz = []
    if 'xy' in planes or 'yx' in planes:
        dih_xy = dihedral_angles(normals_v, plane='XY')
    if 'xz' in planes or 'zx' in planes:
        dih_xz = dihedral_angles(normals_v, plane='XZ')
    if 'yz' in planes or 'zy' in planes:
        dih_yz = dihedral_angles(normals_v, plane='YZ')

    return dih_xy, dih_xz, dih_yz


def clustering(dih_xy, dih_xz, dih_yz=None, method='hist'):
    if dih_yz is None:
        X = np.vstack((dih_xy, dih_xz)).T
    else:
        X = np.vstack((dih_xy, dih_xz, dih_yz)).T
# def clustering(dih_xy, dih_xz, dih_yz):
#     X = np.vstack((dih_xy, dih_xz, dih_yz)).T
    if method in ['kmeans', 'cmeans']:
        n_clusters = 10
        kmeans = sklclu.KMeans(n_clusters=n_clusters)
        kmeans.fit(X)
        labels = kmeans.labels_
        # hist, bin_edges = np.histogram(labels, bins=n_clusters, density=True)
        hist, bins = skiexp.histogram(labels, nbins=n_clusters)
        max_lab = labels[np.argmax(hist)]
        max_labels = labels == max_lab

    elif method == 'hist':
        eps = 4

        hist_xy, bins_xy = skiexp.histogram(dih_xy, nbins=180)
        hist_xz, bins_xz = skiexp.histogram(dih_xz, nbins=180)

        # hist = cv2.calcHist([(dih_xy + 90).astype(np.uint8), (dih_xz + 90).astype(np.uint8)], [0, 1], None, [180, 180], [0, 180, 0, 180])
        # plt.figure()
        # plt.imshow(hist, interpolation='nearest')
        # plt.show()

        # plt.figure()
        # plt.plot(bins_xy+90, hist_xy+90, 'g-', linewidth=3)
        # plt.plot(bins_xz+90, hist_xz+90, 'b-', linewidth=3)
        # plt.show()

        peak_xy = bins_xy[np.argmax(hist_xy)]
        peak_xz = bins_xz[np.argmax(hist_xz)]

        max_labels_xy = (peak_xy - eps < dih_xy) * (dih_xy < peak_xy + eps)
        max_labels_xz = (peak_xz - eps < dih_xz) * (dih_xz < peak_xz + eps)
        max_labels = max_labels_xy * max_labels_xz
    else:
        print 'WARNING! Wrong method type. Using \'hist\' as default.'
        max_labels = clustering(dih_xy, dih_xz)

    return max_labels


def fitting_plane(vertices, faces, max_labels):
    f_sets, v_sets = splitMesh(vertices, faces, mask_v=max_labels)
    n_sets = v_sets.max() + 1
    sizes = np.zeros(n_sets)
    for i in range(n_sets):
        sizes[i] = (v_sets == i).sum()
    max_set = np.argmax(sizes)

    plane = planefit(vertices[v_sets == max_set,...])# * np.array([-1, -1, -1, 1])
    vertices = plane_as_xy_transform(vertices, plane, max_labels)
    if (vertices[:, 2] < 100).sum() > (vertices[:, 2] > 100).sum():
        vertices[:, 2] *= -1

    vertices = align_xy_axes(vertices, max_labels, show=False)

    vertices[:, 1] -= vertices[:, 1].min()

    return vertices


def feet_segmentation(vertices, faces):
    thresh_z = 5
    mask = vertices[:, 2] > thresh_z
    f_sets, v_sets = splitMesh(vertices, faces, mask_v=mask)
    hist, bins = skiexp.histogram(v_sets[v_sets > -1], nbins=v_sets.max() + 1)
    sort_inds = np.argsort(hist)
    mean_x_1 = vertices[v_sets == bins[sort_inds[-1]], 0].mean()
    mean_x_2 = vertices[v_sets == bins[sort_inds[-2]], 0].mean()
    if mean_x_1 < mean_x_2:
        foot_l_mask = v_sets == bins[sort_inds[-1]]
        foot_r_mask = v_sets == bins[sort_inds[-2]]
        foot_l = vertices[foot_l_mask, :]
        foot_r = vertices[foot_r_mask, :]
    else:
        foot_l_mask = v_sets == bins[sort_inds[-2]]
        foot_r_mask = v_sets == bins[sort_inds[-1]]
        foot_l = vertices[foot_l_mask, :]
        foot_r = vertices[foot_r_mask, :]
    feet_mask = foot_l_mask + 2 * foot_r_mask

    return foot_l, foot_l_mask, foot_r, foot_r_mask, feet_mask


# def save_figures(fig_dir, name,  vertices, faces, feet_mask, foot_l_mask, foot_r_mask, max_h,
#                  heel_cut_l, heel_cut_r, pts_l, pts_r, mean_pt_l, mean_pt_r,
#                  foot_l, foot_r, poly1_l, poly2_l, poly1_r, poly2_r, left_points, right_points):
def save_figures(fig_dir, name,  vertices, faces, feet_mask, foot_l_mask, foot_r_mask, max_h,
                 heel_cut_l, heel_cut_r, foot_l, foot_r, poly1_l, poly2_l, poly1_r, poly2_r, left_points, right_points):
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    pt_1_l = left_points[0]
    pt_1_r = right_points[0]
    pt_2_l = left_points[1]
    pt_2_r = right_points[1]
    pt_3_l = left_points[2]
    pt_3_r = right_points[2]
    pt_4_l = left_points[3]
    pt_4_r = right_points[3]

    # generating and saving figure with labeled FEETS
    mlab.clf()
    mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces, scalars=feet_mask.astype(np.int))
    mlab.view(-90, 90)
    mlab.savefig(os.path.join(fig_dir, name) + '_feet.png')

    # generating and saving figure with labeled HEELS
    mlab.clf()
    heel_l_idx = (vertices[:, 1] <= heel_cut_l) * (vertices[:, 2] <= max_h) * foot_l_mask
    heel_r_idx = (vertices[:, 1] <= heel_cut_r) * (vertices[:, 2] <= max_h) * foot_r_mask
    heels_idxs = heel_l_idx + 2 * heel_r_idx
    mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces, scalars=heels_idxs)

    # for i in range(len(pts_l)):
    #     mlab.points3d(pts_l[i][0], pts_l[i][1], pts_l[i][2], color=(0, 0, 0), scale_factor=2)
    # mlab.points3d(mean_pt_l[0], mean_pt_l[1], mean_pt_l[2], color=(1, 0, 1), scale_factor=4)
    #
    # for i in range(len(pts_r)):
    #     mlab.points3d(pts_r[i][0], pts_r[i][1], pts_r[i][2], color=(0, 0, 0), scale_factor=2)
    # mlab.points3d(mean_pt_r[0], mean_pt_r[1], mean_pt_r[2], color=(1, 0, 1), scale_factor=4)

    mlab.view(-60, 90)
    mlab.savefig(os.path.join(fig_dir, name) + '_heel_r.png')
    mlab.view(210, 90)
    mlab.savefig(os.path.join(fig_dir, name) + '_heel_l.png')

    # generating and saving figure with POINTS
    mlab.clf()
    feet_idxs = feet_mask + 3 * heel_l_idx + 4 * heel_r_idx
    mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces, scalars=feet_idxs)

    mlab.points3d(pt_1_l[0], pt_1_l[1], pt_1_l[2], color=(1, 1, 1), scale_factor=8)
    mlab.points3d(pt_1_r[0], pt_1_r[1], pt_1_r[2], color=(1, 1, 1), scale_factor=8)

    mlab.points3d(pt_2_l[0], pt_2_l[1], pt_2_l[2], color=(0, 0, 0), scale_factor=8)
    mlab.points3d(pt_2_r[0], pt_2_r[1], pt_2_r[2], color=(0, 0, 0), scale_factor=8)

    mlab.points3d(pt_3_l[0], pt_3_l[1], pt_3_l[2], color=(0, 1, 1), scale_factor=8)
    mlab.points3d(pt_3_r[0], pt_3_r[1], pt_3_r[2], color=(0, 1, 1), scale_factor=8)

    mlab.points3d(pt_4_l[0], pt_4_l[1], pt_4_l[2], color=(1, 0, 1), scale_factor=8)
    mlab.points3d(pt_4_r[0], pt_4_r[1], pt_4_r[2], color=(1, 0, 1), scale_factor=8)

    mlab.view(-90, 90)
    mlab.savefig(os.path.join(fig_dir, name) + '_points.png')

    # ploting lines
    fig_l = draw_lines(foot_l, poly1_l, poly2_l, left_points)
    fig_l.savefig(os.path.join(fig_dir, name) + '_lines_L.png')
    fig_r = draw_lines(foot_r, poly1_r, poly2_r, right_points)
    fig_r.savefig(os.path.join(fig_dir, name) + '_lines_R.png')

    plt.close('all')


def compare_angles(fname1, fname2, cmp_fname='/home/tomas/Data/Paty/results.xlsx'):
    cviceni = ['augustynova', 'babjak', 'barcala', 'bartovsky', 'brezina', 'cislerova', 'cerveny', 'culik', 'danek_adam',
               'danek_kuba', 'gerlicky', 'hofman', 'houska', 'kordik', 'kralovec', 'kudrnova', 'lastovicka',
               'lhotakova', 'limkova']
    tejp = ['bezdek', 'fischer', 'kasparek', 'kasparkova', 'kopecky', 'pluhar', 'prokyskova', 'rambala', 'reiser', 'resik',
            'samesova', 'svoboda', 'sykorova', 'simsa_jakub', 'simsa_vojtech', 'simsova', 'skalout', 'stajf',
            'sula', 'svarc', 'winklerova', 'zahorik']
    out = ['duchkova', 'klementova']

    angles_1 = pickle.load(open(fname1, 'rb'))
    angles_2 = pickle.load(open(fname2, 'rb'))
    n_names = len(angles_1.keys())
    n_feet = 2 * n_names
    n_cvic = len(cviceni)
    n_tejp = len(tejp)

    # Create a workbook and add a worksheet.
    workbook = xlsxwriter.Workbook(cmp_fname)
    worksheet = workbook.add_worksheet()

    red_format = workbook.add_format({'bg_color': 'red'})
    green_format = workbook.add_format({'bg_color': 'lime'})

    row = 2
    col = 0

    header = ['jmeno', 'zari L', 'zari R', 'rijen L', 'rijen R', '', 'typ', 'diff L', 'diff R', 'lepsi L', 'lepsi_R']
    for (i, h) in enumerate(header):
        worksheet.write(0, i, h)

    lepsi_L = [0, 0]
    lepsi_R = [0, 0]
    horsi_L = [0, 0]
    horsi_R = [0, 0]
    lepsi_obe = [0, 0]
    horsi_obe = [0, 0]
    zlepseni_L = [[], []]
    zlepseni_R = [[], []]
    zhorseni_L = [[], []]
    zhorseni_R = [[], []]
    # Write data
    for (i, name) in enumerate(angles_1.keys()):
        zariL = angles_1[name]['zari'][0]
        zariR = angles_1[name]['zari'][1]
        rijenL = angles_2[name]['rijen'][0]
        rijenR = angles_2[name]['rijen'][1]

        if name in cviceni:
            type_ind = 0
            type = 'cvic'
        elif name in tejp:
            type_ind = 1
            type = 'tejp'
        else:
            type = '?'

        diff_L = rijenL - zariL
        diff_R = rijenR - zariR
        if diff_L >= 0:
            lepsi_L[type_ind] += 1
            zlepseni_L[type_ind].append(diff_L)
        else:
            horsi_L[type_ind] += 1
            zhorseni_R[type_ind].append(diff_L)
        if diff_R >= 0:
            lepsi_R[type_ind] += 1
            zlepseni_R[type_ind].append(diff_R)
        else:
            horsi_R[type_ind] += 1
            zhorseni_R[type_ind].append(diff_R)
        if diff_L >= 0 and diff_R >= 0:
            lepsi_obe[type_ind] += 1
        if diff_L <= 0 and diff_R <= 0:
            horsi_obe[type_ind] += 1

        rowdata = [name, zariL, zariR, rijenL, rijenR, '', type, diff_L, diff_R, diff_L >= 0, diff_R >= 0]

        for (i, d) in enumerate(rowdata):
            worksheet.write(row, i, d)
        row += 1

    lepsi = [lepsi_L[0] + lepsi_R[0], lepsi_L[1] + lepsi_R[1]]
    horsi = [horsi_L[0] + horsi_R[0], horsi_L[1] + horsi_R[1]]
    zlepseni = [zlepseni_L[0] + zlepseni_R[0], zlepseni_L[1] + zlepseni_R[1]]
    zhorseni = [zhorseni_L[0] + zhorseni_R[0], zhorseni_L[1] + zhorseni_R[1]]

    # writing statistics
    row += 2
    row_start = row
    worksheet.write(row, 0, 'CVICENI')
    row += 1
    # worksheet.write(row, 0, 'zlepsilo L [%]')
    # worksheet.write(row, 1, float(zlepsilo_L[0]) / n_cvic * 100, green_format)
    # worksheet.write(row, 2, 'zhorsilo L [%]')
    # worksheet.write(row, 3, float(zhorsilo_L[0]) / n_cvic * 100, red_format)
    # row += 1
    # worksheet.write(row, 0, 'zlepsilo R [%]')
    # worksheet.write(row, 1, float(zlepsilo_R[0]) / n_cvic * 100, green_format)
    # worksheet.write(row, 2, 'zhorsilo R [%]')
    # worksheet.write(row, 3, float(zhorsilo_R[0]) / n_cvic * 100, red_format)
    # row += 1
    # worksheet.write(row, 0, 'zlepsilo obe [%]')
    # worksheet.write(row, 1, float(zlepsilo_obe[0]) / n_cvic * 100)
    # worksheet.write(row, 2, 'zhorsilo obe [%]')
    # worksheet.write(row, 3, float(zhorsilo_obe[0]) / n_cvic * 100)
    # row += 1

    worksheet.write(row, 0, 'lepsi [%]')
    worksheet.write(row, 1, lepsi[0] / (2 * n_cvic) * 100, green_format)
    worksheet.write(row, 2, 'horsi [%]')
    worksheet.write(row, 3, horsi[0] / (2 * n_cvic) * 100, red_format)
    row += 1
    worksheet.write(row, 0, 'prum. zlepseni [%s]' % u'\u00B0')
    worksheet.write(row, 1, np.array(zlepseni[0]).sum() / len(zlepseni[0]))
    worksheet.write(row, 2, 'prum. zhorseni [%s]' % u'\u00B0')
    worksheet.write(row, 3, np.array(zhorseni[0]).sum() / len(zhorseni[0]))

    row = row_start
    worksheet.write(row, 6, 'TEJP')
    row += 1
    # worksheet.write(row, 6, 'zlepsilo L [%]')
    # worksheet.write(row, 7, float(zlepsilo_L[1]) / n_tejp * 100, green_format)
    # worksheet.write(row, 8, 'zhorsilo L [%]')
    # worksheet.write(row, 9, float(zhorsilo_L[1]) / n_tejp * 100, red_format)
    # row += 1
    # worksheet.write(row, 6, 'zlepsilo R [%]')
    # worksheet.write(row, 7, float(zlepsilo_R[1]) / n_tejp * 100, green_format)
    # worksheet.write(row, 8, 'zhorsilo R [%]')
    # worksheet.write(row, 9, float(zhorsilo_R[1]) / n_tejp * 100, red_format)
    # row += 1
    # worksheet.write(row, 6, 'zlepsilo obe [%]')
    # worksheet.write(row, 7, float(zlepsilo_obe[1]) / n_tejp * 100)
    # worksheet.write(row, 8, 'zhorsilo obe [%]')
    # worksheet.write(row, 9, float(zhorsilo_obe[1]) / n_tejp * 100)
    worksheet.write(row, 6, 'lepsi [%]')
    worksheet.write(row, 7, lepsi[1] / (2 * n_tejp) * 100, green_format)
    worksheet.write(row, 8, 'horsi [%]')
    worksheet.write(row, 9, horsi[1] / (2 * n_tejp) * 100, red_format)
    row += 1
    worksheet.write(row, 6, 'prum. zlepseni [%s]' % u'\u00B0')
    worksheet.write(row, 7, np.array(zlepseni[1]).sum() / len(zlepseni[0]))
    worksheet.write(row, 8, 'prum. zhorseni [%s]' % u'\u00B0')
    worksheet.write(row, 9, np.array(zhorseni[1]).sum() / len(zhorseni[0]))

    workbook.close()



def run(fname, save_data=False, save_fig=False, show=False):
    vertices, faces, normals_v = read_ply(fname)
    if faces.shape[1] == 4:
        faces = faces[:, 1:]

    dirs = fname.split('/')
    name = dirs[-1][:-4]
    data_dir = os.path.join('/'.join(dirs[:-1]), 'npy')
    fig_dir = os.path.join('/'.join(dirs[:-1]), 'figs')

    ## vertices, desk_labels = align_with_desk(vertices)

    ## align_xy_axes(vertices, faces, np.ones(vertices.shape[0], dtype=np.bool), n_comps=3, show=True)

    ## align_xy_axes(vertices, faces, desk_labels, n_comps=2, show=True)

    # mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces)
    # mlab.show()

    # DIHEDRAL ANGLES ----------------------------------------------
    print 'Calculating dihedral angles ...',
    # dih_xy = dihedral_angles(normals_v, plane='XY')
    # dih_xz = dihedral_angles(normals_v, plane='XZ')
    # dih_yz = dihedral_angles(normals_v, plane='YZ')
    # dih_xy, dih_xz, dih_yz = get_angles(normals_v)
    dih_xy, dih_xz, dih_yz = get_angles(normals_v, planes=['xy', 'xz'])
    # dih_xy, dih_xz, dih_yz = get_angles(normals_v, planes=['xy', 'xz', 'yz'])
    print 'done'

    # CLUSTERING -------------------------------------------------------
    print 'Clustering ...',
    max_labels = clustering(dih_xy, dih_xz)
    # max_labels = clustering(dih_xy, dih_xz, dih_yz, method='kmeans')
    print 'done'
    # mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces, scalars=max_labels.astype(np.int))
    # mlab.show()

    # FITTING PLANE ------------------------------------------------
    print 'Fitting plane ...',
    vertices = fitting_plane(vertices, faces, max_labels)
    print 'done'

    # mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces, scalars=max_labels.astype(np.int))
    # mlab.show()

    # SEGMENTING FEET ----------------------------------
    print 'Segmenting feet ...',
    foot_l, foot_l_mask, foot_r, foot_r_mask, feet_mask = feet_segmentation(vertices, faces)
    print 'done'

    # CALCULATING MEDIAL AXES OF THE FEET ----------------------------------------
    # axis_l = np.mean(foot_l, 0)
    # axis_r = np.mean(foot_r, 0)

    # foot_ll_mask = (vertices[:, 0] <= axis_l[0]) * (vertices[:, 1] < axis_l[1]) * foot_l_mask
    # foot_lr_mask = (vertices[:, 0] > axis_l[0]) * (vertices[:, 1] < axis_l[1]) * foot_l_mask
    # foot_rl_mask = (vertices[:, 0] <= axis_r[0]) * (vertices[:, 1] < axis_r[1]) * foot_r_mask
    # foot_rr_mask = (vertices[:, 0] > axis_r[0]) * (vertices[:, 1] < axis_r[1]) * foot_r_mask
    # feet_sides_mask = foot_ll_mask + 2 * foot_lr_mask + 3 * foot_rl_mask + 4 * foot_rr_mask

    # PT-1 ... LYTKO = CALF -------------------------------------------------------
    print 'Finding calf points ...',
    calf_pt_l = calf_point(foot_l)
    calf_pt_r = calf_point(foot_r)
    print 'done'

    # PT-2 ... ACHILOVKA -----------------------------------------------------------
    print 'Finding Achilleus\' points ...',
    achill_pt_l = achill_point(foot_l, 'l', ankle_type='outer')
    achill_pt_r = achill_point(foot_r, 'r', ankle_type='outer')
    print 'done'

    # PT-3 & 4 ... HEEL POINTS ------------------------------------------------------
    print 'Finding heel points ...',
    # min_h = 20
    max_h = 100
    # ---- old ----
    # heel_cut_l, pts_l, mean_pt_l = cut_heel(foot_l, 'l', max_h)
    # closest_pt_l, widest_pt_l = heel_points(foot_l, heel_cut_l, max_h, show=False)
    # heel_cut_r, pts_r, mean_pt_r = cut_heel(foot_r, 'r', max_h)
    # closest_pt_r, widest_pt_r = heel_points(foot_r, heel_cut_r, max_h, show=False)

    # ---- new ----
    heel_cut_l = achill_pt_l[1]
    heel_cut_r = achill_pt_r[1]
    widest_pt_l = widest_point(foot_l, heel_cut_l, max_h, show=False, show_now=False)
    widest_pt_r = widest_point(foot_r, heel_cut_r, max_h, show=False)

    under_ankle_pt_l, band_l = under_ankle_point(foot_l, 'l')
    under_ankle_pt_r, band_r = under_ankle_point(foot_r, 'r')

    pt_1_l = calf_pt_l
    pt_1_r = calf_pt_r
    pt_2_l = achill_pt_l
    pt_2_r = achill_pt_r
    pt_3_l = under_ankle_pt_l
    pt_3_r = under_ankle_pt_r
    pt_4_l = widest_pt_l
    pt_4_r = widest_pt_r

    # mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces)
    # for pt in band_l:
    #     mlab.points3d(pt[0], pt[1], pt[2], color=(0, 0, 0), scale_factor=2)
    # for pt in band_r:
    #     mlab.points3d(pt[0], pt[1], pt[2], color=(0, 0, 0), scale_factor=2)
    # mlab.points3d(under_ankle_pt_l[0], under_ankle_pt_l[1], under_ankle_pt_l[2], color=(1, 0, 1), scale_factor=4)
    # mlab.points3d(under_ankle_pt_r[0], under_ankle_pt_r[1], under_ankle_pt_r[2], color=(1, 0, 1), scale_factor=4)
    # mlab.show()

    # cut_l, pts_band_l, pt_cut_l = under_ankle_cut(foot_l, 'l')
    # cut_r, pts_band_r, pt_cut_r = under_ankle_cut(foot_r, 'r')

    # mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces)
    # for pt in pts_band_l:
    #     mlab.points3d(pt[0], pt[1], pt[2], color=(0, 0, 0), scale_factor=2)
    # for pt in pts_band_r:
    #     mlab.points3d(pt[0], pt[1], pt[2], color=(0, 0, 0), scale_factor=2)
    # mlab.points3d(pt_cut_l[0], pt_cut_l[1], pt_cut_l[2], color=(1, 0, 1), scale_factor=4)
    # mlab.points3d(pt_cut_r[0], pt_cut_r[1], pt_cut_r[2], color=(1, 0, 1), scale_factor=4)
    # mlab.show()


    # heel_l_idx = (vertices[:, 1] <= heel_cut_l) * (vertices[:, 2] <= max_h) * foot_l_mask
    # heel_r_idx = (vertices[:, 1] <= heel_cut_r) * (vertices[:, 2] <= max_h) * foot_r_mask
    # heels_idxs = heel_l_idx + 2 * heel_r_idx
    # mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces, scalars=heels_idxs)
    # mlab.show()
    # mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces)
    # mlab.points3d(mean_pt_l[0], mean_pt_l[1], mean_pt_l[2], color=(1, 0, 1), scale_factor=4)
    # mlab.points3d(mean_pt_r[0], mean_pt_r[1], mean_pt_r[2], color=(1, 0, 1), scale_factor=4)
    # mlab.show()

    print 'done'

    # left_points = [calf_l, achill_l, closest_pt_l, widest_pt_l]
    # right_points = [calf_r, achill_r, closest_pt_r, widest_pt_r]
    left_points = [pt_1_l, pt_2_l, pt_3_l, pt_4_l]
    right_points = [pt_1_r, pt_2_r, pt_3_r, pt_4_r]

    # ANGLE CALCULATION -------------------------------------------------------------
    print 'Calculating angles ...'
    theta_l, poly1_l, poly2_l = angle(np.array(left_points))
    theta_r, poly1_r, poly2_r = angle(np.array(right_points))
    print '\t angle L = %.1f' % theta_l
    print '\t angle R = %.1f' % theta_r
    print 'done'

    if save_data:
        print 'Saving data ...',
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        np.save(os.path.join(data_dir, name) + '_vertices.npy', vertices)
        np.save(os.path.join(data_dir, name) + '_foot_l_mask.npy', foot_l_mask)
        np.save(os.path.join(data_dir, name) + '_foot_r_mask.npy', foot_r_mask)
        np.save(os.path.join(data_dir, name) + '_feet_mask.npy', feet_mask)
        np.save(os.path.join(data_dir, name) + '_foot_l.npy', foot_l)
        np.save(os.path.join(data_dir, name) + '_foot_r.npy', foot_r)
        np.save(os.path.join(data_dir, name) + '_faces.npy', faces)
        np.save(os.path.join(data_dir, name) + '_left_points.npy', left_points)
        np.save(os.path.join(data_dir, name) + '_right_points.npy', right_points)
        print 'done'

    if save_fig:
        print 'Saving figures ...',
        if not os.path.exists(fig_dir):
            os.mkdir(fig_dir)
        # save_figures(fig_dir, name, vertices, faces, feet_mask, foot_l_mask, foot_r_mask, max_h,
        #              heel_cut_l, heel_cut_r, pts_l, pts_r, mean_pt_l, mean_pt_r,
        #              foot_l, foot_r, poly1_l, poly2_l, poly1_r, poly2_r, left_points, right_points)
        save_figures(fig_dir, name, vertices, faces, feet_mask, foot_l_mask, foot_r_mask, max_h,
                     heel_cut_l, heel_cut_r, foot_l, foot_r, poly1_l, poly2_l, poly1_r, poly2_r, left_points, right_points)
        print 'done'

    if show:
        mlab.clf()
        heel_l_idx = (vertices[:, 1] <= heel_cut_l) * (vertices[:, 2] <= max_h) * foot_l_mask
        heel_r_idx = (vertices[:, 1] <= heel_cut_r) * (vertices[:, 2] <= max_h) * foot_r_mask
        feet_idxs = feet_mask + 3 * heel_l_idx + 4 * heel_r_idx

        mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces, scalars=feet_idxs)
        # mesh_vis = mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces, scalars=(vertices[:, 2] > thresh_z).astype(np.int))
        # mesh_vis = mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces, scalars=feet_mask.astype(np.int))
        # mlab.colorbar()

        # for i in range(len(pts_l)):
        #     mlab.points3d(pts_l[i][0], pts_l[i][1], pts_l[i][2], color=(0, 0, 0), scale_factor=2)
        # mlab.points3d(mean_pt_l[0], mean_pt_l[1], mean_pt_l[2], color=(1, 0, 1), scale_factor=4)
        #
        # for i in range(len(pts_r)):
        #     mlab.points3d(pts_r[i][0], pts_r[i][1], pts_r[i][2], color=(0, 0, 0), scale_factor=2)
        # mlab.points3d(mean_pt_r[0], mean_pt_r[1], mean_pt_r[2], color=(1, 0, 1), scale_factor=4)

        mlab.points3d(pt_1_l[0], pt_1_l[1], pt_1_l[2], color=(1, 1, 1), scale_factor=8)
        mlab.points3d(pt_1_r[0], pt_1_r[1], pt_1_r[2], color=(1, 1, 1), scale_factor=8)

        mlab.points3d(pt_2_l[0], pt_2_l[1], pt_2_l[2], color=(0, 0, 0), scale_factor=8)
        mlab.points3d(pt_2_r[0], pt_2_r[1], pt_2_r[2], color=(0, 0, 0), scale_factor=8)

        mlab.points3d(pt_3_l[0], pt_3_l[1], pt_3_l[2], color=(0, 1, 1), scale_factor=8)
        mlab.points3d(pt_3_r[0], pt_3_r[1], pt_3_r[2], color=(0, 1, 1), scale_factor=8)

        mlab.points3d(pt_4_l[0], pt_4_l[1], pt_4_l[2], color=(1, 0, 1), scale_factor=8)
        mlab.points3d(pt_4_r[0], pt_4_r[1], pt_4_r[2], color=(1, 0, 1), scale_factor=8)

        mlab.show()

        plt.show()

    print '\n'

    return theta_l, theta_r


def batch_processing(dir, month, save_data, dave_fig, show):
    names = tools.get_names(dir)
    # names = ['fischer', 'augustynova']

    n_files = len(names)

    log_file = open('log_file.txt', 'w')

    angles = dict()
    failed = list()
    processed = 0
    for (i, name) in enumerate(names):
        print '--  Processing file %i/%i - %s  --' % (i + 1, n_files, name + '.ply')
        fname = os.path.join(dir[:-1], name + '.ply')
        # try:
        theta_L, theta_R = run(fname, save_data=save_data, save_fig=save_fig, show=show)
        angles[name] = {month: (theta_L, theta_R)}

        log_file.write(name + '.ply ... ok\n')
        processed += 1

    if save_data:
        fname = os.path.join(dir, 'npy', 'angles_' + month + '.p')
        pickle.dump(angles, open(fname, 'wb'))
        # np.save(os.path.join(dir, 'npy', 'angles_' + month + '.npy'), angles)
        # np.save(os.path.join(dir, 'npy', 'failed.npy'), failed)

    log_file.close()

    print '\n------------------'
    print 'DONE'
    print 'processed: %i/%i' % (processed, n_files)

    print 'angles:'
    print angles

    print 'failed:'
    print failed


# ---------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    warnings.filterwarnings('error')

    save_data = True
    save_fig = True
    show = False

    months = ['zari', 'rijen']

    # fname = '/home/tomas/Data/Paty/new/zari/cislerova.ply'
    # fname = '/home/tomas/Data/Paty/new/rijen/kasparkova.ply'
    # fname = '/home/tomas/Data/Paty/new/zari/augustynova.ply'
    # theta_L, theta_R = run(fname, save_data=save_data, save_fig=save_fig, show=show)

    # for month in months:
    #     print '\n -----  %s  -----' % month
    #     dir = '/home/tomas/Data/Paty/new/%s/' % month
    #     batch_processing(dir, month, save_data, save_fig, show)
    #
    print 'Writing results to file...',
    # fname1 = '/home/tomas/Data/Paty/zari/ply/npy/angles_zari.p'
    # fname2 = '/home/tomas/Data/Paty/rijen/ply/npy/angles_rijen.p'
    fname1 = '/home/tomas/Data/Paty/new/zari/npy/angles_zari.p'
    fname2 = '/home/tomas/Data/Paty/new/rijen/npy/angles_rijen.p'
    resname = '/home/tomas/Data/Paty/new/results.xlsx'
    compare_angles(fname1, fname2, cmp_fname=resname)
    print 'done'

    alert()