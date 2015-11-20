__author__ = 'tomas'

import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import pyplot
from mayavi import mlab
# from tvtk.api import tvtk
# from mayavi.scripts import mayavi2
# from mayavi.sources.vtk_data_source import VTKDataSource
# from mayavi.modules.surface import Surface

from stl import mesh

def read_ply(fname):
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

    return vertices, normals, faces

def read_stl(fname):
    # model = mesh.Mesh.from_file(fname)
    class Model():
        def __init__(self):
            pass
    model = Model()
    model.v0 = np.array([[0, 1, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [0, 0, 0], [0, 0, 0]])
    model.v1 = np.array([[1, 0, 1], [0, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 0], [0, 0, 1]])
    model.v2 = np.array([[0, 0, 1], [1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 1]])

    model.vectors = np.zeros((6, 3, 3))
    # Top of the cube
    model.vectors[0, :, :] = np.array([[0, 1, 1], [1, 0, 1], [0, 0, 1]])
    model.vectors[1, :, :] = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 1]])
    # Right face
    model.vectors[2, :, :] = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0]])
    model.vectors[3, :, :] = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 0]])
    # Left face
    model.vectors[4, :, :] = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1]])
    model.vectors[5, :, :] = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 1]])

    model.normals = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0]])

    vertices = np.vstack((model.v0, model.v1, model.v2))
    tmp = np.ascontiguousarray(vertices).view(np.dtype((np.void, vertices.dtype.itemsize * vertices.shape[1])))
    _, idx = np.unique(tmp, return_index=True)
    vertices = vertices[idx]
    faces = np.zeros((model.vectors.shape[0], 3), dtype=np.int)
    for i in range(faces.shape[0]):
        for pt in range(3):
            faces[i, pt] = np.argwhere((vertices == model.vectors[i, pt, :]).sum(1) == 3)

    # computing normals
    normals_f = model.normals.copy()
    normals_v = np.zeros((vertices.shape[0], 3))

    for i in range(vertices.shape[0]):
        norms = model.normals[np.nonzero((faces == i).sum(1))[0], :]
        normals_v[i, :] = np.mean(norms, 0)

    return vertices, faces, normals_v, normals_f

def view(model):
    from mayavi.sources.vtk_data_source import VTKDataSource
    from mayavi.modules.surface import Surface
    from mayavi.api import Engine

    src = VTKDataSource(data=model)
    # mayavi.add_source(src)
    # s = Surface()
    # mayavi.add_module(s)

    # Create the MayaVi engine and start it.
    engine = Engine()
    engine.start()
    scene = engine.new_scene()
    engine.add_source(src)

    # Add Surface Module
    surface = Surface()
    engine.add_module(surface)

    from pyface.api import GUI
    gui = GUI()
    gui.start_event_loop()

if __name__ == '__main__':
    fname = '/home/tomas/Data/Paty/zari/augustynova.stl'
    # fname = '/home/tomas/Data/Paty/zari/augustynova.ply'
    # vertices, normals, faces = read_ply(fname)
    vertices, faces, normals_v, normals_f = read_stl(fname)

    # model = tvtk.PolyData(points=vertices, polys=faces)
    # model.point_data.vectors = normals_v

    # view(model)

    if faces.shape[1] == 4:
        faces = faces[:, 1:]
    mesh_vis = mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces)
    # mlab.mesh(model.x, model.y, model.z)
    # mlab.points3d(model.x.flatten(), model.y.flatten(), model.z.flatten(), scale_factor=0.6)

    # cell_data = mesh_vis.mlab_source.dataset.cell_data
    # cell_data.vectors = normals_f
    # cell_data.vectors.name = 'Cell data'
    # cell_data.update()
    #
    # mesh2 = mlab.pipeline.set_active_attribute(mesh_vis, cell_vectors='Cell data')
    # mlab.pipeline.surface(mesh2)

    point_data = mesh_vis.mlab_source.dataset.point_data
    point_data.scalars = normals_v
    point_data.scalars.name = 'Point data'
    point_data.update()

    mesh2 = mlab.pipeline.set_active_attribute(mesh_vis, point_scalars='Point data')
    mlab.pipeline.surface(mesh2)


    mlab.show()

    # # Create a new plot
    # figure = pyplot.figure()
    # axes = mplot3d.Axes3D(figure)
    # axes.add_collection3d(mplot3d.art3d.Poly3DCollection(model.vectors))
    # # Auto scale to the mesh size
    # scale = model.points.flatten(-1)
    # axes.auto_scale_xyz(scale, scale, scale)
    #
    # # Show the plot to the screen
    # pyplot.show()