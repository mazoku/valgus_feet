__author__ = 'tomas'

import numpy as np
# from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from mayavi import mlab
# from tvtk.api import tvtk
# from mayavi.scripts import mayavi2
# from mayavi.sources.vtk_data_source import VTKDataSource
# from mayavi.modules.surface import Surface

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


def align_with_desk(vertices):
    cloud = pcl.PointCloud(vertices.astype(np.float32))
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

    labels = np.zeros(vertices.shape[0])
    labels[indices] = 1

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


if __name__ == '__main__':

    # fname = '/home/tomas/Data/Paty/zari/ply/augustynova.stl'
    fname = '/home/tomas/Data/Paty/zari/ply/augustynova.ply'
    vertices, faces, normals_v = read_ply(fname)
    vertices_o = vertices.copy()

    vertices, desk_labels = align_with_desk(vertices)

    # counting dihedral angles between XY-plane and vertex normals
    # dihedrals = dihedral_angles(normals_v)
    # hist, bin_edges = np.histogram(dihedrals, bins=180, density=True)
    # bins = [np.mean((bin_edges[x], bin_edges[x + 1])) for x in range(len(hist)) ]
    #
    # max_ang = bins[np.argmax(hist)]
    # print max_ang
    # plt.figure()
    # plt.plot(bins, hist)
    # plt.show()

    # labels = np.zeros(vertices.shape[0])
    # labels = np.where((max_ang - 5 < dihedrals) * (dihedrals < max_ang + 5), 1, 0)
    # labels = np.abs(dihedrals - max_ang)

    if faces.shape[1] == 4:
        faces = faces[:, 1:]
    # mesh_vis = mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces)
    mesh_vis = mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces, scalars=desk_labels)
    # mesh_vis = mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces, scalars=vertices[:, 2])
    # mesh_vis = mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces, scalars=range(vertices.shape[0]))

    mlab.show()