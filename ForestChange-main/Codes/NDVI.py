from osgeo import gdal
import matplotlib.pyplot as plt

def GetNDVI(FilePathInList,FilePathOutList):
    assert len(FilePathInList)==len(FilePathOutList) , 'File list not equal'
    for i in range(len(FilePathInList)):
        FilePathIn=FilePathInList[i]
        FilePathOut=FilePathOutList[i]
        podataset = gdal.Open(FilePathIn, gdal.GA_ReadOnly)
        bandNIR = podataset.GetRasterBand(4)
        bandRED = podataset.GetRasterBand(3)
        XSize = podataset.RasterXSize
        YSize = podataset.RasterYSize
        dataNIR = bandNIR.ReadAsArray(0, 0, XSize, YSize)
        dataRED = bandRED.ReadAsArray(0, 0, XSize, YSize)
        NIR = dataNIR.astype(float)  # 转化成float类型，整数的话负数要报错
        RED = dataRED.astype(float)  # 转化成float类型，整数的话负数要报错
        ndvi = (NIR-RED)/(NIR+RED)
        poDriver = gdal.GetDriverByName('GTiff')
        pwDataset = poDriver.Create(FilePathOut, XSize, YSize, 1, gdal.GDT_Float32)
        pwDataset.SetGeoTransform(podataset.GetGeoTransform())
        pwDataset.SetProjection(podataset.GetProjection())
        pwDataset.GetRasterBand(1).WriteArray(ndvi)
        pwDataset = None
        podataset = None
        # ndvi是numpyArray数组
        plt.imshow(ndvi, cmap='Greens')
        plt.show()
        plt.close()

FilePathInList=[]
for i in range(1,4):
    for j in range(1,5):
        FilePathInList.append('zone/00%d202%d0.TIF'%(j,i))
FilePathOutList=[i.replace('.TIF','_ndvi.tif').replace('zone','zone/ndvi') for i in FilePathInList]
GetNDVI(FilePathInList,FilePathOutList)