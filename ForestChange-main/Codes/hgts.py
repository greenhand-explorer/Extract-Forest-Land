from osgeo import gdal
import os
import matplotlib.pyplot as plt


def HGTS2TIF(InputFile, OutputFile, ShowPic):
    """
    将hgts转变为高程
    InputFile：输入文件
    OutputFIle：输出文件
    ShowPic：是否预览结果文件
    """
    # 打开.hgts数据
    dataset = gdal.Open(InputFile, gdal.GA_ReadOnly)
    # 读取高程数据
    elevation_data = dataset.GetRasterBand(1).ReadAsArray()
    # 获取地理坐标信息
    geo_transform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    # 创建GeoTIFF文件
    driver = gdal.GetDriverByName("GTiff")
    output_file = driver.Create(
        OutputFile, elevation_data.shape[1], elevation_data.shape[0], 1, gdal.GDT_Float32)
    # 设置地理坐标信息
    output_file.SetGeoTransform(geo_transform)
    output_file.SetProjection(projection)
    # 写入高程数据
    output_file.GetRasterBand(1).WriteArray(elevation_data)
    if (ShowPic):
        plt.imshow(elevation_data)
        plt.show()
    # 关闭文件
    output_file = None
    dataset = None


InFileDir = r'E:\Big_Files\3S实践项目数据\数据下载\安徽省雷达数据'
OutFileDir = r'E:\Big_Files\3S实践项目数据\数据处理\雷达数据转高程'
for i in range(29, 35):
    for j in range(115, 119):
        Infile = os.path.join(InFileDir, 'n%de%d.hgts' % (i, j))
        Outfile = os.path.join(OutFileDir, '安徽省%dN,%dE高程数据.tif' % (i, j))
        if (os.path.exists(Infile)):
            HGTS2TIF(Infile, Outfile, 0)
        else:
            print('缺少数据%dN,%dE' % (i, j))
