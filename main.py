__author__ = 'tomas'

import numpy as np
# from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from mayavi import mlab
# from tvtk.api import tvtk
# from mayavi.scripts import mayavi2
# from mayavi.sources.vtk_data_source import VTKDataSource
# from mayavi.modules.surface import Surface
import transformations as trans

from sklearn import decomposition as skldec
from sklearn import cluster as sklclu
from skimage import exposure as skiexp

from stl import mesh
import pcl


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


def dihedral_angles(vec, plane='XY'):
    print 'Calculating dihedral angles ...',
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
        dih = np.arcsin(np.dot(n, vec[i,:]) / (np.linalg.norm(n) * np.linalg.norm(vec[i,:])))
        dih = np.degrees(dih)
        dih = min(dih, 180 - dih)
        dihs[i] = dih

    print 'done'
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
    labeled_v = vertices[labels, ...]

    min_en = np.Inf
    min_en_rot = None
    for i in range(0, 180, 1):
        TM = trans.axangle2mat((0, 0, 1), np.deg2rad(i))
        pts_t = np.array(np.dot(labeled_v, TM.T))

        en = get_energy(pts_t)
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
        print 'Min energy: %.2f, rotation angle:%i degs' % (min_en, min_en_rot
                                                   )
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

    en = 10000 / std_x + 10000 / std_y + len_x + len_y

    if show:
        plt.figure()
        plt.subplot(121), plt.plot(pts[:, 0], pts[:, 1], 'bx')
        plt.subplot(222), plt.plot(bins_x, hist_x, 'b-'), plt.title('X-projection')
        plt.subplot(224), plt.plot(bins_y, hist_y, 'b-'), plt.title('Y-projection')
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


# ---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    fname = '/home/tomas/Data/Paty/zari/ply/augustynova.ply'
    # fname = '/home/tomas/Data/Paty/zari/ply/babjak.ply'
    # fname = '/home/tomas/Data/Paty/zari/ply/barcala.ply'
    vertices, faces, normals_v = read_ply(fname)
    if faces.shape[1] == 4:
        faces = faces[:, 1:]
    vertices_o = vertices.copy()

    # vertices, desk_labels = align_with_desk(vertices)

    # align_xy_axes(vertices, faces, np.ones(vertices.shape[0], dtype=np.bool), n_comps=3, show=True)

    # align_xy_axes(vertices, faces, desk_labels, n_comps=2, show=True)

    # counting dihedral angles between XY-plane and vertex normals
    # dihedrals = dihedral_angles(normals_v)
    # hist, bin_edges = np.histogram(dihedrals, bins=180, density=True)
    # bins = [np.mean((bin_edges[x], bin_edges[x + 1])) for x in range(len(hist)) ]
    # max_ang = bins[np.argmax(hist)]

    # print max_ang
    # plt.figure()
    # plt.plot(bins, hist)
    # plt.show()
    dih_xy = dihedral_angles(normals_v, plane='XY')
    dih_xz = dihedral_angles(normals_v, plane='XZ')
    dih_yz = dihedral_angles(normals_v, plane='YZ')

    # KMEANS
    print 'KMeans clustering ...',
    X = np.vstack((dih_xy, dih_xz, dih_yz)).T
    n_clusters = 10
    kmeans = sklclu.KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    labels = kmeans.labels_
    # hist, bin_edges = np.histogram(labels, bins=n_clusters, density=True)
    hist, bins = skiexp.histogram(labels, nbins=n_clusters)
    max_lab = labels[np.argmax(hist)]
    max_labels = labels == max_lab
    print 'done'

    # FITTING PLANE
    print 'Fitting plane ...',
    f_sets, v_sets = splitMesh(vertices, faces, mask_v=max_labels)
    n_sets = v_sets.max() + 1
    sizes = np.zeros(n_sets)
    for i in range(n_sets):
        sizes[i] = (v_sets == i).sum()
    max_set = np.argmax(sizes)

    # plane = planefit(vertices[max_labels,...])# * np.array([-1, -1, -1, 1])
    plane = planefit(vertices[v_sets == max_set,...])# * np.array([-1, -1, -1, 1])
    vertices = plane_as_xy_transform(vertices, plane, max_labels)
    if (vertices[:, 2] < 100).sum() > (vertices[:, 2] > 100).sum():
        vertices[:, 2] *= -1

    # pca(vertices, faces, max_labels, n_comps=3, show=False)
    vertices = align_xy_axes(vertices, max_labels, show=False)

    # labels = np.zeros(vertices.shape[0], dtype=np.bool)
    # labels = np.where((max_ang - 5 < dihedrals) * (dihedrals < max_ang + 5), 1, 0)
    # labels = np.abs(dihedrals - max_ang)

    # vertices, desk_labels = align_with_desk(vertices, idxs=labels>0)
    # vertices, desk_labels = align_with_desk(vertices, idxs=max_labels)
    print 'done'

    # SEGMENTING FEET ----------------------------------
    print 'Segmenting feet ...',
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
    print 'done'

    # mesh_vis = mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces, scalars=(vertices[:, 2] > thresh_z).astype(np.int))
    # mesh_vis = mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces, scalars=v_sets.astype(np.int))
    mesh_vis = mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces, scalars=feet_mask.astype(np.int))
    # mesh_vis = mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces, scalars=vertices[:, 2])
    # mesh_vis = mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces, scalars=max_labels.astype(np.int))
    # mesh_vis = mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces, scalars=desk_labels.astype(np.int))
    mlab.colorbar()

    mlab.show()