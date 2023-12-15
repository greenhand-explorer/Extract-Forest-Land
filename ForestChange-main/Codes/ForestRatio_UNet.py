from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"]=False

def HowManyForest(input_file, input_filer, forest_code, nodata_code=65535):
    """统计林地占比
    
    input_file：输入文件
    forest_code：标签中是林地的数值
    nodata_code：设置为nodata的数值
    dpi：分辨率
    返回值：林地占比，林地面积（单位：平方千米）
    """
    dataset=gdal.Open(input_file,gdal.GA_ReadOnly)
    datasetr=gdal.Open(input_filer,gdal.GA_ReadOnly)
    XSize=dataset.RasterXSize
    YSize=dataset.RasterYSize
    dataset.GetRasterBand(1).SetNoDataValue(nodata_code)
    dpix=datasetr.GetGeoTransform()[1]
    dpiy=abs(datasetr.GetGeoTransform()[5])
    array=dataset.ReadAsArray(0,0,XSize,YSize)
    dataset=None
    forest_pixel=np.count_nonzero(array==forest_code)
    forest_area=dpix*dpiy*forest_pixel/(1000**3)
    return forest_area

fig,ax=plt.subplots(3,3,figsize=(8,8))
for area in range(1,4):
    area_list=[]
    for year in range(2021,2024):
        File='Codes/UNet/test_imgs_output/00%d%d0_pre.png'%(area,year)
        dataset=gdal.Open(File,gdal.GA_ReadOnly)
        XSize=dataset.RasterXSize
        YSize=dataset.RasterYSize
        array=dataset.ReadAsArray(0,0,XSize,YSize)
        ax[area-1, year-2021].imshow(array,cmap='Greens')
        ax[area-1, year-2021].set_title('区域%d %d年林地状况'%(area,year))
        dataset=None
        ax[area-1, year-2021].get_xaxis().set_visible(False)
        ax[area-1, year-2021].get_yaxis().set_visible(False)       
plt.show()
plt.close()

year_list=[2021,2022,2023]
fig,ax=plt.subplots(1,3,figsize=(9,3))
for area in range(1,4):
    area_list=[]
    for year in range(2021,2024):
        File='Codes/UNet/test_imgs_output/00%d%d0_pre.png'%(area,year)
        FileR='zone/Classified/00%d%d0_classified.tif'%(area,year)
        farea=HowManyForest(File,FileR,1)
        area_list.append(farea)
    ax[area-1].set_xticks(year_list)
    ax[area-1].tick_params(axis='y')
    ax[area-1].plot(year_list,area_list,label='林地面积/平方千米')
    ax[area-1].set_title('区域%d 林地面积变化图'%area)
    ax[area-1].legend()
fig.tight_layout()
plt.show()
plt.close()
