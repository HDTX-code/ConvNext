# opencv实现图像旋转实例
import cv2
import random
import matplotlib.pylab as plt

# 定义旋转函数
import numpy as np


def ImageRotate(image):
    image = cv2.resize(image, (224, 224))
    height, width = image.shape[:2]  # 输入(H,W,C)，取 H，W 的zhi
    center = (width / 2, height / 2)  # 绕图片中心进行旋转
    angle = random.randint(-180, 180)  # 旋转方向取（-180，180）中的随机整数值，负为逆时针，正为顺势针
    scale = 1  # 将图像缩放为80%

    # 获得旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, scale)
    # 进行仿射变换，边界填充为255，即白色，默认为黑色
    image_rotation = cv2.warpAffine(src=image, M=M, dsize=(height, width), borderValue=(255, 255, 255))

    return image_rotation


def ImageNew(src):
    blur_img = cv2.GaussianBlur(src, (0, 0), 5)
    usm = cv2.addWeighted(src, 1.5, blur_img, -0.5, 0)
    result = usm
    return result


def Image_GaussianBlur(img):
    kernel_size = (5, 5)
    sigma = 1.5
    img = cv2.GaussianBlur(img, kernel_size, sigma)
    return img


def Data_strength(path, num, opt):
    image = cv2.imread(path)
    if num == 1:
        image = ImageRotate(image)
    elif num == 2:
        image = ImageRotate(image)
    elif num == 3:
        image = ImageRotate(image)
    elif num == 4:
        image = ImageRotate(image)
    elif num == 5:
        image = ImageNew(ImageRotate(image))
    elif num == 6:
        image = ImageRotate(ImageNew(image))
    elif num == 7:
        image = Image_GaussianBlur(ImageRotate(image))
    elif num == 8:
        image = ImageRotate(ImageRotate(image))
    elif num == 9:
        image = ImageRotate(ImageRotate(ImageRotate(image)))
    elif num == 10:
        image = ImageNew(ImageRotate(ImageRotate(ImageRotate(image))))
    elif num == 11:
        image = Image_GaussianBlur(ImageRotate(ImageRotate(ImageRotate(image))))
    img_0 = cv2.resize(image, (opt.w, opt.h))
    img1 = np.zeros([3, opt.w, opt.h])  # 改变img1的时候不改变img
    img1[0, :, :] = img_0[:, :, 2]
    img1[1, :, :] = img_0[:, :, 1]
    img1[2, :, :] = img_0[:, :, 0]  # cv2读取的是bgr,转换成rgb就要做一下变通
    return img1