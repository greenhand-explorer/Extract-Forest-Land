from osgeo import gdal
import numpy as np
size=4595
driver = gdal.GetDriverByName("GTiff")
for i in range(2021,2024):
    dataset=gdal.Open('zone/002%d0.tif'%i)
    array=dataset.ReadAsArray(0,0,dataset.RasterXSize,dataset.RasterYSize)
    dataset=None
    array=array[:,:size,:size]
    output=driver.Create('zone/002%d0.tif'%i, size, size, 4, gdal.GDT_UInt16)
    output.WriteArray(array)
    output=None

    dataset=gdal.Open('Codes/UNet/test_imgs/002%d0.tif'%i)
    array=dataset.ReadAsArray(0,0,dataset.RasterXSize,dataset.RasterYSize)
    dataset=None
    array=array[:,:size,:size]
    output=driver.Create('Codes/UNet/test_imgs/002%d0.tif'%i, size, size, 4, gdal.GDT_UInt16)
    output.WriteArray(array)
    output=None