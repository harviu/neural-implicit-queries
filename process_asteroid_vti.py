from vtkmodules import all as vtk
from vtkmodules.util import numpy_support
import numpy as np
# from torch.nn.functional import interpolate
# import torch



# reader = vtk.vtkXMLImageDataReader()
# reader.SetFileName('../data/ethanediol.vti')
# reader.Update()
# data = reader.GetOutput()
# np_data = numpy_support.vtk_to_numpy(data.GetPointData().GetArray('log(s)')).reshape(115,116,134)
# data = np.zeros(np_data.shape, np.float32)
# data[:] = np_data[:]
# data.squeeze().tofile('../data/ethanediol.bin')

# reader = vtk.vtkXMLImageDataReader()
# reader.SetFileName('../data/pv_insitu_500x500x500_28516.vti')
# reader.Update()
# data = reader.GetOutput()
# print(data)
# np_data = numpy_support.vtk_to_numpy(data.GetPointData().GetArray('v02')).reshape(500,500,500)
# np_data.squeeze().tofile('../data/99_500_v02.bin')
# data = torch.from_numpy(np_data)[None,None,:,:,:]
# print(data.min(), data.max())
# resampled = interpolate(data,512,mode='trilinear')
# print(resampled.min(), resampled.max())
# resampled.numpy().squeeze().tofile('../data/99_512_v02.bin')
# vtk_data = vtk.vtkImageData()
# vtk_data.SetSpacing((9218.44, 5611.22, 4809.62))
# vtk_data.SetOrigin((-2.3e+06, -500000, -1.2e+06))
# vtk_data.SetExtent((0, 512, 0, 512, 0, 512))
# vtk_data.SetDimensions((513, 513, 513))
# vtk_array = numpy_support.numpy_to_vtk(v)
# vtk_array.SetName(k)
# pd.AddArray(vtk_array)
# vtk_data.GetPointData().SetArray('v02')
# print(vtk_data)