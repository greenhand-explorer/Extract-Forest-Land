from osgeo import gdal
import numpy as np
# 输入遥感影像路径
input_paths = ["数据处理/00220210.TIF"]

for input_path in input_paths:
    # 打开输入遥感影像
    input_ds = gdal.Open(input_path)

    # 获取输入遥感影像的波段数和影像尺寸
    num_bands = input_ds.RasterCount
    image_width = input_ds.RasterXSize
    image_height = input_ds.RasterYSize
    data = input_ds.ReadAsArray(0,0,image_width,image_height)

    driver = gdal.GetDriverByName("GTiff")
    # 前四个波段的影像
    output_path='实验数据-深度学习和标注/'+input_path.split('/')[-1]
    output_ds = driver.Create(output_path, image_width, image_height, 4, gdal.GDT_UInt16)
    for i in range(4):
        output_ds.GetRasterBand(i+1).SetNoDataValue(65535)
    banddata=data[:4, :, :]
    output_ds.WriteArray(banddata)
    output_ds.SetProjection(input_ds.GetProjection())
    output_ds.SetGeoTransform(input_ds.GetGeoTransform())
    output_ds=None    
    # 最后一个是标签
    output_path='实验数据-深度学习和标注/'+input_path.split('/')[-1].replace('.TIF','_label.tif')
    output_ds = driver.Create(output_path, image_width, image_height, 1, gdal.GDT_Byte)
    banddata=data[4]
    banddata[banddata==65535]=0
    output_ds.WriteArray(banddata)
    output_ds.SetProjection(input_ds.GetProjection())
    output_ds.SetGeoTransform(input_ds.GetGeoTransform())
    input_ds=None
    output_ds=None

# 关闭资源
input_ds = None
output_ds = None