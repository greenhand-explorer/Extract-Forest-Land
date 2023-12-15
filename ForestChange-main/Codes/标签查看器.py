from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

Image.MAX_IMAGE_PIXELS = None


data = Image.open(r'实验数据-深度学习和标注\myj_label.tif')
data = np.array(data)

print(data.shape)
print('最大值：%d' % np.max(data))
print('最小值：%d' % np.min(data))
print('1占比：%f' % (np.count_nonzero(data)/(data.shape[0]*data.shape[1])))
plt.imshow(data)
plt.show()
plt.close()
