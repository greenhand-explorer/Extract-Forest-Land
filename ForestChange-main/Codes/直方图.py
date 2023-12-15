from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt


def plot_histogram(image_path):
    # 打开遥感影像文件
    dataset = gdal.Open(image_path)

    # 获取波段数
    num_bands = dataset.RasterCount

    # 创建一个列表用于存储各波段数据
    band_data = []

    for i in range(1, num_bands + 1):
        # 读取影像数据
        band = dataset.GetRasterBand(i)
        data = band.ReadAsArray()

        # 将数据存入列表
        band_data.append(data)

    # 将列表转换为NumPy数组，便于处理
    band_data = np.array(band_data)

    # 将波段数据展平
    flattened_data = band_data.flatten()

    # 绘制直方图
    plt.hist(flattened_data, bins=256, range=(0, 128), color='b', alpha=0.7)

    # 设置图表标题和坐标轴标签
    plt.title('Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    # 显示直方图
    plt.show()

    # 关闭文件
    dataset = None


# 调用示例
image_path = '实验数据-深度学习和标注\\高光谱数据8位.tif'
plot_histogram(image_path)
