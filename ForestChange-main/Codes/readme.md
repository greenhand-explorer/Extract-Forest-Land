# 更新：目前总结出来的几点

1. 分类标签必须要用0和1，不能用1和2（二分类一般来说都是0和1），用1和2甚至更大的数值一方面容易报错，另一方面会分类出0
2. 使用最新数据训练和预测
3. 不能使用自动标注，必须要把所有的林地全部标注出来且林地必须是一个完整的整体。shl第一幅图和自动标注得到的结果是零散的，识别出来的林地也是零散的
4. 确保影像和标签一一对应再去训练
5. 林地占比需要足够大。可以使用数据增广来实现

# UNet代码运行注意事项

## 文件夹目录设置

1.在最顶级目录（git克隆下来之后的那个文件夹）打开vscode

2.保证“实验数据-深度学习和标注”文件夹中有shl.TIF,shl_label.tif,myj.TIF,myj_label.tif影像。带Label的是标签，不带的是影像。

3.保证Codes文件夹下有UNet文件夹，UNet文件夹下有models_16bit,test_imgs,test_imgs_output文件夹

## 运行UNet

打开UNet.py，点击运行

是否启用影像筛选：一般来说要，相当于把林地多的进行数据增广，剩下的保留一份。如果不启用的话就是所有数据各一份

损失函数一律选2（FotalLoss）

是否使用早停：早停是指训练到最后验证精度反而下降时直接停止训练。一般来说选是，如果想看为什么到后面精度下降可以选否

然后读取数据

再输入：总共训练几次，每多少次验证一次，输入影像波段数，输出图像类别数

一般总共训练10或者100或者200（看速度），每多少次随意，输入波段数4，输出类别数2

然后就开始跑了

跑出来的模型在models_16bit文件夹里

最后会输出训练Loss的图，训练Accuracy的图和验证Accuracy的图。如果是训练一次验证一次的，第二个和第三个一起出现

## 全图预测

**先把UNet.py中class UNet那个类和下面的构造函数、forwarda函数全部复制到UNetModel.py中，替换那里面的UNet类**

test_imgs中放入你想要预测的高光谱影像

然后运行whole_image_predict.py

选用第几个模型：在models_16bit中有epoch几.pkl，你要用哪个就输入序号

输出结果在test_imgs_output中，可以用ArcGIS Pro打开结果

输出影像类别数：2

## 修改网络和参数

FocalLoss参数：在Train函数那边找到match case那里改

学习率：Train函数下面for epoch in tqdm(range(epoch_num))下一行的lr

数据集：找到_dataset类定义的下面，传入数据集的有两个数组，第一个是影像数组，第二个是标签数组，影像和标签要一一对应。例如shl.tif和myj.tif就要对应后面标签数组的shl_label.tif和myj_label.tif

验证精度反而下降多少次之后早停：Val函数最后面Early_Stop_Counter>20的20可以调整

数据增广幅度：_dataset类print('数据增广...')下面for i in range(100)那个100，有两个，要同时改且要改成一样的

修改UNet：

```python
class UNet(nn.Module):
    ''' UNet(3-level): downconv *3 => conv_conv => upconv *3 => conv[1*1]'''

    def __init__(self, in_channels, out_channels, starting_filters=32, bn_momentum=0.1):
        super(UNet, self).__init__()
        self.conv1 = downconv(in_channels, starting_filters, bn_momentum)
        self.conv2 = downconv(
            starting_filters, starting_filters * 2, bn_momentum)
        self.conv3 = downconv(
            starting_filters * 2, starting_filters * 4, bn_momentum)

	# 添加层：self.convk=downconv(
        #   starting_filters * 上面一层右边的, starting_filters * 上面一层右边的两倍, bn_momentum)

        self.convconv = conv_conv(starting_filters * 上面一层右边的,
                                  starting_filters * 上面一层右边的两倍, bn_momentum)

	# 添加层：self.convk=downconv(
        #   starting_filters * 上面一层右边的, starting_filters * 上面一层右边的一半, bn_momentum)

        self.upconv3 = upconv_concat(
            starting_filters * 8, starting_filters * 4, bn_momentum)
        self.upconv2 = upconv_concat(
            starting_filters * 4, starting_filters * 2, bn_momentum)
        self.upconv1 = upconv_concat(
            starting_filters * 2, starting_filters, bn_momentum)
        self.conv_out = nn.Conv2d(
            starting_filters, out_channels, kernel_size=1, padding=0, stride=1)

    def forward(self, X):
        X, conv1 = self.conv1(X)
        X, conv2 = self.conv2(X)
        X, conv3 = self.conv3(X)
	# 下降层就按照上面写
        X = self.convconv(X)
	# 上升层就按照下面写
        X = self.upconv3(X, conv3)
        X = self.upconv2(X, conv2)
        X = self.upconv1(X, conv1)
        X = self.conv_out(X)
        return X
```

改完记得把UNet.py中class UNet那个类和下面的构造函数、forwarda函数全部复制到UNetModel.py中，替换那里面的UNet类，否则全图预测会出错
