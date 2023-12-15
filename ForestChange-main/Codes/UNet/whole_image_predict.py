# 规则预测+随机预测
import torch
import numpy as np
from PIL import Image
import os
from osgeo import gdal

from UNetModel import *


class test_IMG():
    def __init__(self, model, TestImg, class_num, Pre_Folder, norm=True, randomtime=0):

        print("---test_IMG---")
        print(TestImg.split('\\')[-1])
        dataset = self.readTif(TestImg)
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        imgdata = dataset.ReadAsArray(0, 0, width, height)
        imgdata = np.float32(imgdata)
        if norm:
            imgdata = self.data_normal(imgdata)
        imgdata = torch.from_numpy(imgdata)
        imgdata = imgdata.unsqueeze(0)

        # 类别数
        self.class_num = class_num
        # 宽高
        self.img_h = height
        self.img_w = width

        # 新建容器，用于统计各类别数量
        self.predect_count_list = []
        for classes in range(self.class_num):
            self.predect_count_list.append(
                np.zeros((self.img_h, self.img_w)).astype(np.uint8))  # 记录每个类的个数

        name = os.path.basename(TestImg)
        name = name.split('.')[0]
        predict_path = Pre_Folder + "\\" + name + '_pre.png'
        self.test_patches(imgdata, model, predict_path, 128, randomtime)

    def readTif(self, fileName):
        img_tif = gdal.Open(fileName)
        if img_tif == None:
            print(fileName + "File Open Failure")
            return

        return img_tif

    def data_normal(self, feature_files):
        for i in range(feature_files.shape[0]):
            tile = feature_files[i]

            d_min = tile.min()
            d_max = tile.max()

            dst = d_max - d_min
            if dst == 0:
                feature_files[i] = tile
                continue

            feature_files[i] = (tile - d_min) / dst

        return feature_files

    def test_patches(self, test_img, model_img, predict_path, patch_size, run_time):

        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")  # gpu
        print(device)
        img_model = model_img
        img_model = img_model.to(device)
        test_img = test_img.to(device)
        img_model.eval()

        img_w = self.img_w
        img_h = self.img_h

        # 规则裁剪 重叠度50%
        cur_h = 0
        while cur_h < img_h:
            start_h = cur_h
            end_h = cur_h + patch_size
            if end_h >= img_h:
                end_h = img_h
                start_h = end_h - patch_size

            cur_w = 0
            while cur_w < img_w:
                start_w = cur_w
                end_w = cur_w + patch_size
                if end_w >= img_w:
                    end_w = img_w
                    start_w = end_w - patch_size
                self.test_single_patch(
                    test_img, start_h, end_h, start_w, end_w, img_model)

                cur_w += int(patch_size / 2)
            cur_h += int(patch_size / 2)

        # 随机裁剪，次数run_time
        for i in range(run_time):
            start_h = np.random.randint(0, img_h - 1)
            end_h = start_h + patch_size
            if end_h >= img_h:
                end_h = img_h - 1
                start_h = end_h - patch_size
            start_w = np.random.randint(0, img_w - 1)
            end_w = start_w + patch_size
            if end_w >= img_w:
                end_w = img_w - 1
                start_w = end_w - patch_size
            self.test_single_patch(
                test_img, start_h, end_h, start_w, end_w, img_model)

        # 类别
        image_predect = np.array([self.predect_count_list[i]
                                 for i in range(self.class_num)])
        image_predect = np.argmax(image_predect, 0).astype(np.uint8)
        # img_rgb = self.cvtRGB(image_predect)
        image_predect = Image.fromarray(image_predect)
        image_predect.save(predict_path)
        return True

    def test_single_patch(self, test_img, start_h, end_h, start_w, end_w, model_img):
        with torch.no_grad():

            img_patch = test_img[:, :, start_h:end_h, start_w:end_w]

            img_outputs = model_img(img_patch)
            self.agg_vote(img_outputs, start_h, end_h, start_w, end_w)

    def agg_vote(self, outputs, start_h, end_h, start_w, end_w):
        _, max_index = torch.max(outputs, 1)
        max_index = max_index.cpu().detach().numpy()
        max_index = max_index.reshape((max_index.shape[1], max_index.shape[2]))

        for classes in range(self.class_num):
            class_index = (max_index == classes)
            self.predect_count_list[classes][start_h:end_h,
                                             start_w:end_w][class_index] += 1


if __name__ == "__main__":
    # 文件目录需要修改
    # 参数
    model_num = int(input('选用第几个模型：'))
    img_dir = 'Codes/UNet/test_imgs'     # 待预测图像文件夹（整图，非裁剪）
    model_path = 'Codes/UNet/models_16bit/model_epoch%d.pkl' % model_num  # 模型路径
    pred_dir = 'Codes/UNet/test_imgs_output'   # 与预测图像输出文件夹
    class_num = int(input('请输入分类后影像的类别数：'))       # 类别数
    bnorm = False      # 是否归一化：这取决于训练模型时图像是否归一化
    randomtime = 100  # 随机预测的次数

    img_model = torch.load(model_path)
    for img in os.listdir(img_dir):
        test_IMG(img_model, img_dir+"\\"+img,
                 class_num, pred_dir, bnorm, randomtime)
