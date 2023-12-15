import rasterio
from rasterio.plot import show
# 打开哨兵二号遥感影像高光谱数据1.tif
sentinel2 = rasterio.open('实验数据-深度学习和标注\\高光谱数据1.tif')
# 获取红光波段和近红外波段
red = sentinel2.read(3)
nir = sentinel2.read(4)
# 计算NDVI
ndvi = (nir - red) / (nir + red)
# 显示NDVI图像
show(ndvi, cmap='Greens_r')

# 获取哨兵2号遥感影像的元数据
meta = sentinel2.meta
# 更新元数据以添加新的波段
meta.update(count=meta['count'] + 1)
# 创建一个新的tif文件，其中包含原始哨兵二号遥感影像和新的NDVI波段
with rasterio.open('结果数据-深度学习和标注\\带NDVI的高光谱影像.tif', 'w', **meta) as dst:
    for i in range(1, meta['count']):
        dst.write(sentinel2.read(i), i)
        dst.write(ndvi.astype(rasterio.float32), meta['count'])
