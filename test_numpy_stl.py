from __future__ import division

import numpy as np
from stl import mesh

# Create 3 faces of a cube
data = np.zeros(6, dtype=mesh.Mesh.dtype)

# Top of the cube
data['vectors'][0] = np.array([[0, 1, 1], [1, 0, 1], [0, 0, 1]])
data['vectors'][1] = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 1]])
# Right face
data['vectors'][2] = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0]])
data['vectors'][3] = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 0]])
# Left face
data['vectors'][4] = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1]])
data['vectors'][5] = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 1]])

model = mesh.Mesh(data.copy())
pass