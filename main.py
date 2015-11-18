__author__ = 'tomas'

import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import pyplot
from mayavi import mlab

from stl import mesh

if __name__ == '__main__':
    fname = '/home/tomas/Data/Paty/zari/augustynova.stl'
    model = mesh.Mesh.from_file(fname)

    # f=open(fname, 'r')
    #
    # x=[]
    # y=[]
    # z=[]
    #
    # for line in f:
    #     strarray=line.split()
    #     if strarray[0]=='vertex':
    #         x=append(x,double(strarray[1]))
    #         y=append(y,double(strarray[2]))
    #         z=append(z,double(strarray[3]))
    #
    # triangles=[(i, i+1, i+2) for i in range(0, len(x),3)]
    #
    # pass
    # The mesh normals (calculated automatically)
    # model.normals

    # The mesh vectors
    # model.v0, model.v1, model.v2

    # Accessing individual points (concatenation of v0, v1 and v2 in triplets)
    # assert (model.points[0][0:3] == model.v0[0]).all()
    # assert (model.points[0][3:6] == model.v1[0]).all()
    # assert (model.points[0][6:9] == model.v2[0]).all()
    # assert (model.points[1][0:3] == model.v0[1]).all()

    vertices = np.array(vertices)
    faces = np.array(faces)

    mlab.triangular_mesh(vertices[:,0], vertices[:,1], vertices[:,2], faces)
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