import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataset as Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from osgeo import gdal
import random
# 统计训练次数和验证次数
Train_Time = 0
Val_Time = 0
Early_Stop_Counter = 0
Loss_Training = []
Accuracy_Train = []
Accuracy_Val = []
print('输入：是否启用影像筛选：1-使用，0-不使用 ')
IsUseChoose = int(input())
print('输入：损失函数选择 1-CrossEntropyError 2-FocalLoss')
LossFunctionChoose = int(input())
print('输入：是否使用早停：1-使用，0-不使用 ')
IsEarlyStop = int(input())


def HowManyForest(array) -> float:
    """
    计算二维数组中非0值占总值的比例
    林地像素占切片总像素的多少？返回这个值
    """
    return np.count_nonzero(array)/(array.shape[0]*array.shape[1])


class FocalLoss(nn.Module):
    """
    FocalLoss损失函数
    """

    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()

# 自定义数据集类


class _dataset(Dataset.Dataset):
    def __init__(self, Train_images, Label_images, bnorm=False, Image_Size=256, IsUseChoose=True):
        self.TrainList = Train_images
        self.LabelList = Label_images
        self.bnorm = bnorm
        self.size = 0
        self.transform = transforms.ToTensor()
        self.TrainArray = []
        self.LabelArray = []
        self.images = []
        self.labels = []

        assert len(self.TrainList) == len(
            self.LabelList), 'Images and Labels have different num'

        print('读取数据...')
        # 读取影像数据
        for idx in range(len(self.TrainList)):
            # 读取一张影像
            image = self.TrainList[idx]
            Train_image = gdal.Open(image)
            XSize1 = Train_image.RasterXSize
            YSize1 = Train_image.RasterYSize
            self.TrainArray.append(
                Train_image.ReadAsArray(0, 0, XSize1, YSize1))
            Train_image = None

            # 读取一张影像
            label = self.LabelList[idx]
            Label_image = gdal.Open(label)
            XSize2 = Label_image.RasterXSize
            YSize2 = Label_image.RasterYSize
            self.LabelArray.append(
                Label_image.ReadAsArray(0, 0, XSize2, YSize2))
            Label_image = None

            assert XSize1 == XSize2 and YSize1 == YSize2, 'Size Not Equal'

        print('裁剪...')
        # 裁剪影像到images批和Labels批
        for idx in range(len(self.TrainArray)):
            # 存放每个切片是否符合要求的数组
            IsEnoughForestIndex = -1
            IsEnoughForestList = []
            ####################### 标签 ##############################
            label = self.LabelArray[idx]
            label = np.array(label)
            array_shape = label.shape
            # 定义裁剪的长宽和重叠率
            crop_size = Image_Size
            overlap = 0.5
            # 计算裁剪的步长
            step = int(crop_size * (1 - overlap))
            # 进行裁剪
            for i in range(0, array_shape[0] - crop_size + 1, step):
                for j in range(0, array_shape[1] - crop_size + 1, step):
                    # 切片操作进行裁剪
                    crop = label[i:i+crop_size, j:j+crop_size]
                    # 如果启用影像筛选
                    if (IsUseChoose == True):
                        # 林地像素占比要大于50%
                        if (HowManyForest(crop) > 0.3):
                            IsEnoughForestList.append(True)
                        else:
                            IsEnoughForestList.append(False)
                    else:
                        IsEnoughForestList.append(True)
                    IsEnoughForestIndex += 1
                    # 符合要求的进行数据增广并放入训练集中
                    if (IsEnoughForestList[IsEnoughForestIndex] and IsUseChoose):
                        print('数据增广...')
                        for ii in tqdm(range(1)):
                            self.labels.append(crop)
                            # 旋转90°，180°，270°
                            self.labels.append(
                                np.rot90(crop, k=1, axes=(0, 1)))
                            self.labels.append(
                                np.rot90(crop, k=2, axes=(0, 1)))
                            self.labels.append(
                                np.rot90(crop, k=3, axes=(0, 1)))
                    # 不启用就增加一份
                    else:
                        self.labels.append(crop)
            ####################### 影像 ##############################
            IsEnoughForestIndex = -1
            image = self.TrainArray[idx]
            image = np.array(image)
            array_shape = image.shape
            # 定义裁剪的长宽和重叠率
            crop_size = Image_Size
            overlap = 0.5
            # 计算裁剪的步长
            step = int(crop_size * (1 - overlap))
            # 进行裁剪
            for i in range(0, array_shape[1] - crop_size + 1, step):
                for j in range(0, array_shape[2] - crop_size + 1, step):
                    # 切片操作进行裁剪
                    crop = image[:, i:i+crop_size, j:j+crop_size]
                    IsEnoughForestIndex += 1
                    # 启用影像筛选的情况下，增广符合要求的影像
                    if (IsEnoughForestList[IsEnoughForestIndex] and IsUseChoose):
                        print('数据增广...')
                        for ii in tqdm(range(1)):
                            # 符合要求的影像切片进行数据增广
                            self.images.append(crop)
                            # 旋转90°，180°，270°
                            self.images.append(
                                np.rot90(crop, k=1, axes=(1, 2)))
                            self.images.append(
                                np.rot90(crop, k=2, axes=(1, 2)))
                            self.images.append(
                                np.rot90(crop, k=3, axes=(1, 2)))
                    # 不启用就增加一份
                    else:
                        self.images.append(crop)
        assert len(self.images) == len(self.labels), 'images num != label num'
        self.size = min(len(self.images), len(self.labels))
        print('林地占比：%f' % (sum([np.count_nonzero(self.labels[i]) for i in range(len(
            self.labels))])/(self.labels[0].shape[0]*self.labels[0].shape[1]*len(self.labels))))
        # 检查影像和标签是否对应
        # for i in range(len(self.images)):
        #     poDriver = gdal.GetDriverByName('GTiff')
        #     pwDataset = poDriver.Create('图片检查/影像%d.tif'%i, 128, 128, 4, gdal.GDT_UInt16)
        #     pwDataset.WriteArray(self.images[i])
        #     pwDataset=None
        #     pwDataset = poDriver.Create('图片检查/标签%d.tif'%i, 128, 128, 1, gdal.GDT_UInt16)
        #     pwDataset.WriteArray(self.labels[i])
        #     pwDataset=None


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # idx:批中的序号
        image = np.array(self.images[idx])
        label = np.array(self.labels[idx])
        # 线性映射归一化
        if self.bnorm:
            min_values = np.min(image, axis=(1, 2))
            max_values = np.max(image, axis=(1, 2))
            normalized_array = np.zeros_like(image, dtype=np.float32)
            for i in range(image.shape[0]):
                normalized_array[i] = (
                    image[i] - min_values[i]) / (max_values[i] - min_values[i])
            image = normalized_array
        # 转为tensor形式
        image = np.array(image).astype(np.float32)
        # image = np.transpose(image, [2, 1, 0])
        image = torch.from_numpy(image)

        label = np.array(label).astype(np.float32)
        label = torch.from_numpy(label)
        return image, label
        # image:4*256*256 label:256*256


datatrain = _dataset(['实验数据-深度学习和标注/00220210.TIF'],
                     ['实验数据-深度学习和标注/00220210_label.tif'], IsUseChoose=IsUseChoose, bnorm=False, Image_Size=128)
datatest = _dataset(['实验数据-深度学习和标注/00220210.TIF'],
                     ['实验数据-深度学习和标注/00220210_label.tif'], IsUseChoose=IsUseChoose, bnorm=False, Image_Size=128)

# 定义unet模型相关类


class conv_conv(nn.Module):
    ''' conv_conv: (conv[3*3] + BN + ReLU) *2 '''

    def __init__(self, in_channels, out_channels, bn_momentum=0.1):
        super(conv_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )

    def forward(self, X):
        X = self.conv(X)
        return X


class downconv(nn.Module):
    ''' downconv: conv_conv => maxpool[2*2] '''

    def __init__(self, in_channels, out_channels, bn_momentum=0.1):
        super(downconv, self).__init__()
        self.conv = conv_conv(in_channels, out_channels, bn_momentum)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, X):
        X = self.conv(X)
        pool_X = self.pool(X)
        return pool_X, X


class upconv_concat(nn.Module):
    ''' upconv_concat: upconv[2*2] => cat => conv_conv '''

    def __init__(self, in_channels, out_channels, bn_momentum=0.1):
        super(upconv_concat, self).__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = conv_conv(in_channels, out_channels, bn_momentum)

    def forward(self, X1, X2):
        X1 = self.upconv(X1)
        feature_map = torch.cat((X2, X1), dim=1)
        X1 = self.conv(feature_map)
        return X1


# 层数：4
class UNet(nn.Module):
    ''' UNet(3-level): downconv *3 => conv_conv => upconv *3 => conv[1*1]'''

    def __init__(self, in_channels, out_channels, starting_filters=32, bn_momentum=0.1):
        super(UNet, self).__init__()
        self.conv1 = downconv(in_channels, starting_filters, bn_momentum)
        self.conv2 = downconv(
            starting_filters, starting_filters * 2, bn_momentum)
        self.conv3 = downconv(
            starting_filters * 2, starting_filters * 4, bn_momentum)
        self.conv4 = downconv(
            starting_filters * 4, starting_filters * 8, bn_momentum)
        self.convconv = conv_conv(starting_filters * 8,
                                  starting_filters * 16, bn_momentum)
        # 可以加一个conv5，16，upconv也要加一个对应的。相当于多一个downconv，convconv不变
        self.upconv4 = upconv_concat(
            starting_filters * 16, starting_filters * 8, bn_momentum)
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
        X, conv4 = self.conv4(X)
        X = self.convconv(X)
        X = self.upconv4(X, conv4)
        X = self.upconv3(X, conv3)
        X = self.upconv2(X, conv2)
        X = self.upconv1(X, conv1)
        X = self.conv_out(X)
        return X
# 记得改UNetModel.py


def Train(epoch_num, T, in_channels, out_channels, IsUseChoose=True, LossFunctionChoose=1, IsEarlyStop=True):
    """
    epoch_num是训练总次数
    T是每训练几轮就验证一次
    in_channels输入波段
    out_channels输出类别数
    """
    # 以下为训练部分
    # 5
    # 载入数据
    global Train_Time
    global Val_Time
    global datatrain
    print("train:", len(datatrain))
    train_dataloader = DataLoader(datatrain, batch_size=8, shuffle=True)
    # 使用GPU
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")  # gpu
    print(device)
    # 实例化模型，in_channel输入波段，并导入GPU
    unet = UNet(in_channels=in_channels, out_channels=out_channels)
    unet = unet.to(device)
    # 设置损失函数为交叉熵函数（划掉）FocalLoss，并导入GPU
    if(LossFunctionChoose == 1):
        Loss_function = nn.CrossEntropyLoss()
    elif(LossFunctionChoose == 2):
        Loss_function = FocalLoss(alpha=0.25, gamma=2)
    Loss_function = Loss_function.to(device)
    # 设置优化方法为自适应动量法
    # optimizer = torch.optim.Adam(unet.parameters(), lr=1e-7)
    # scheduler = torch.optim.lr_scheduler() 查一下这个函数的用法，基本上用steplr
    # epoch代表遍历完所有样本的过程，将epoch设置10，即遍历完样本10次
    for epoch in range(epoch_num):
        optimizer = torch.optim.Adam(unet.parameters(), lr=1e-6)
        print('EPOCH %d/%d' % (epoch + 1, epoch_num))
        print('-'*10)
        running_correct_full = 0
        # 批量输入数据
        for images, labels in tqdm(train_dataloader):
            # 导入GPU
            images = images.to(device)
            labels = labels.to(device)
            outputs = unet(images)                          # 将影像输入网络得到输出
            optimizer.zero_grad()                           # 将梯度置为0
            loss = Loss_function(outputs, labels.long())    # 计算损失
            loss.backward()                                 # 损失反向传播
            optimizer.step()                                # 优化更新网络权重\
            _, preds = torch.max(outputs, 1)                        # 预测类别值
            # 统计分类正确像元数
            running_correct_full += torch.sum(preds == labels.data)
        # 训练后直接接入Val()
        # 打印损失值
        print('loss is %f' % (loss.item()))
        Train_Time += 1
        Loss_Training.append(loss.item())
        # 保存模型
        torch.save(unet, 'Codes/UNet/models_16bit/model_epoch%d.pkl' %
                   (epoch + 1))
        acc = running_correct_full.double()/(len(datatrain) *
                                             labels.shape[1]*labels.shape[2])
        print("Training Accuracy: ", acc.cpu().numpy())
        Accuracy_Train.append(acc.cpu().numpy())
        # 如果是需要检查的遍数
        if (epoch + 1) % T == 0:
            IsStop = Val(epoch+1, IsUseChoose=IsUseChoose,
                         IsEarlyStop=IsEarlyStop)
        else:
            IsStop = False
        if (IsStop == True):
            return


def Val(epoch, IsUseChoose=True, IsEarlyStop=True):
    global Train_Time
    global Val_Time
    global Early_Stop_Counter
    # 以下为验证部分
    #########################################################################
    # 载入数据
    print("Val...")

    print("test:", len(datatest))
    # 使用GPU
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")  # gpu
    print(device)
    test_dataloader = DataLoader(datatest, batch_size=8, shuffle=False)
    # 载入训练好的模型
    net = torch.load('Codes/UNet/models_16bit/model_epoch%d.pkl' %
                     epoch, map_location=lambda storage, loc: storage)
    net = net.to(device)
    # 设置为测试模式
    net.eval()
    running_correct_full = 0
    # img_no = 0
    for images, labels in tqdm(test_dataloader):
        # 导入GPU
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)                                   # 得到网络输出
        _, preds = torch.max(outputs, 1)                        # 预测类别值
        running_correct_full += torch.sum(preds == labels.data)   # 统计分类正确像元数
        # 输出结果查看
        images = images.cpu()
        labels = labels.cpu()
        preds = preds.cpu()
        images = images.numpy()
        preds = preds.numpy()
        labels = labels.numpy()
    # 计算测试精度
    acc = running_correct_full.double()/(len(datatest) *
                                         labels.shape[1]*labels.shape[2])
    print("Overall Accuracy: ", acc.cpu().numpy())
    Val_Time += 1
    Accuracy_Val.append(acc.cpu().numpy())
    if (IsEarlyStop == False):
        return False
    else:
        last = len(Accuracy_Val)-1
        if (last > 0 and Accuracy_Val[last] < Accuracy_Val[last-1]):
            Early_Stop_Counter += 1
        if (Early_Stop_Counter > 50):
            print('Accuracy最高的是模型%d' % (Accuracy_Val.index(max(Accuracy_Val))))
            return True


# 文件目录需要修改
if (__name__ == '__main__'):
    print('输入：总共训练几次，每多少次验证一次，输入影像波段数，输出图像类别数')
    TrainTime, T, in_channels, out_channels = map(int, input().split())
    Train(TrainTime, T, in_channels, out_channels,
          IsUseChoose, LossFunctionChoose, IsEarlyStop)
    lst = [i for i in range(1, Train_Time+1)]
    plt.plot(lst, Loss_Training)
    plt.show()
    plt.close()
    if (T == 1):
        plt.plot(lst, Accuracy_Train, label='Train Accuracy')
        plt.plot(lst, Accuracy_Val, label='Val Accuracy')
        plt.legend()
    else:
        plt.plot(lst, Accuracy_Train)
        plt.show()
        plt.close()
        lst = [i for i in range(1, TrainTime//T + 1)]
        plt.plot(lst, Accuracy_Val)
    plt.show()
    plt.close()
