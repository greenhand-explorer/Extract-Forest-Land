from osgeo import gdal
import numpy as np

def DisBlack(Imagefile,Labelfile):
    image=gdal.Open(Imagefile)
    imagearray=image.ReadAsArray(0,0,image.RasterXSize,image.RasterYSize)
    image=None
    label=gdal.Open(Labelfile)
    labelarray=label.ReadAsArray(0,0,label.RasterXSize,label.RasterYSize)
    label=None
    indices = np.argwhere(labelarray != 0)
    x = np.max(indices[:, 0])
    y = np.max(indices[:, 1])
    print(x, y)
    size=min(int(x),int(y))
    driver = gdal.GetDriverByName("GTiff")
    imagearray=imagearray[:,:size,:size]
    labelarray=labelarray[:size,:size]
    output_image=driver.Create(Imagefile, size, size, 4, gdal.GDT_UInt16)
    output_image.WriteArray(imagearray)
    output_image=None
    output_label=driver.Create(Labelfile, size, size, 1, gdal.GDT_Byte)
    output_label.WriteArray(labelarray)
    output_label=None

DisBlack('实验数据-深度学习和标注/00220210.TIF','实验数据-深度学习和标注/00220210_label.tif')