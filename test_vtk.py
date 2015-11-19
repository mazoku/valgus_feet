import vtk

filename = '/home/tomas/Data/Paty/zari/augustynova.stl'

reader = vtk.vtkSTLReader()
reader.SetFileName(filename)

mapper = vtk.vtkPolyDataMapper()
if vtk.VTK_MAJOR_VERSION <= 5:
    mapper.SetInput(reader.GetOutput())
else:
    mapper.SetInputConnection(reader.GetOutputPort())

actor = vtk.vtkActor()
actor.SetMapper(mapper)

# Create a rendering window and renderer
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)

# Create a renderwindowinteractor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

# Assign actor to the renderer
ren.AddActor(actor)

# Enable user interface interactor
iren.Initialize()
renWin.Render()
iren.Start()

# # from __future__ import division
# #
# # import numpy as np
# # from vtk_visualizer import *
# #
# #
# # vtkControl = VTKVisualizerControl()
# #
# # # Generate 1000 random points
# # xyz = np.random.rand(1000,3)*100.0
# #
# # vtkControl.AddPointCloudActor(xyz)
# #
# # # For visualizing points with normals as shaded points:
# # # vtkControl.AddShadedPointsActor(....)
# #
# # # For visualizing points with normals as "hedge hodgs":
# # # vtkControl.AddHedgeHogActor(....)
# #
# # # Render the 3D data (after all components have been added)
# # vtkControl.Render()
# #
# # # Reset the camera position/orientation to fit with the data
# # vtkControl.ResetCamera()
#
# import vtk
# # The colors module defines various useful colors.
# from vtk.util.colors import tomato
#
# # This creates a polygonal cylinder model with eight circumferential
# # facets.
# cylinder = vtk.vtkCylinderSource()
# cylinder.SetResolution(8)
#
# # The mapper is responsible for pushing the geometry into the graphics
# # library. It may also do color mapping, if scalars or other
# # attributes are defined.
# cylinderMapper = vtk.vtkPolyDataMapper()
# cylinderMapper.SetInputConnection(cylinder.GetOutputPort())
#
# # The actor is a grouping mechanism: besides the geometry (mapper), it
# # also has a property, transformation matrix, and/or texture map.
# # Here we set its color and rotate it -22.5 degrees.
# cylinderActor = vtk.vtkActor()
# cylinderActor.SetMapper(cylinderMapper)
# cylinderActor.GetProperty().SetColor(tomato)
# cylinderActor.RotateX(30.0)
# cylinderActor.RotateY(-45.0)
#
# # Create the graphics structure. The renderer renders into the render
# # window. The render window interactor captures mouse events and will
# # perform appropriate camera or actor manipulation depending on the
# # nature of the events.
# ren = vtk.vtkRenderer()
# renWin = vtk.vtkRenderWindow()
# renWin.AddRenderer(ren)
# iren = vtk.vtkRenderWindowInteractor()
# iren.SetRenderWindow(renWin)
#
# # Add the actors to the renderer, set the background and size
# ren.AddActor(cylinderActor)
# ren.SetBackground(0.1, 0.2, 0.4)
# renWin.SetSize(200, 200)
#
# # This allows the interactor to initalize itself. It has to be
# # called before an event loop.
# iren.Initialize()
#
# # We'll zoom in a little by accessing the camera and invoking a "Zoom"
# # method on it.
# ren.ResetCamera()
# ren.GetActiveCamera().Zoom(1.5)
# renWin.Render()
#
# # Start the event loop.
# iren.Start()