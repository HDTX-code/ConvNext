import cv2
import math
import numpy as np

from PIL import Image


def draw_min_rect_circle(img, cnts, opt):  # conts = contours
    img = np.copy(img)
    ii = 0
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 0 and h > 0:
            if ii == 0:
                xmin = x
                xmax = x + w
                ymin = y
                ymax = y + h
                ii = 1
            else:
                if xmin > x:
                    xmin = x
                if xmax < x + w:
                    xmax = x + w
                if ymin > y:
                    ymin = y
                if ymax < y + h:
                    ymax = y + h
    if ii != 0:
        if xmin - (xmax - xmin) * 0.01 >= 0:
            xmin = math.floor(xmin - (xmax - xmin) * 0.01)
        else:
            xmin = 0
        if xmax + (xmax - xmin) * 0.01 <= img.shape[1]:
            xmax = math.floor(xmax + (xmax - xmin) * 0.01)
        else:
            xmax = img.shape[1]
        if ymin - (ymax - ymin) * 0.05 >= 0:
            ymin = math.floor(ymin - (ymax - ymin) * 0.05)
        else:
            ymin = 0
        if ymax + (ymax - ymin) * 0.05 <= img.shape[0]:
            ymax = math.floor(ymax + (ymax - ymin) * 0.05)
        else:
            ymax = img.shape[0]
        cropImg = img[ymin:ymax, xmin:xmax]  # 裁剪图像
        image = cropImg
        # image = Image.fromarray(cropImg).convert('RGB')
        # image = image.resize((opt.w, opt.h), Image.ANTIALIAS)
    else:
        image = img
        # cv2.imshow("image", img)
        # image = Image.fromarray(img).convert('RGB')
        # image = image.resize((opt.w, opt.h), Image.ANTIALIAS)
    return image


def get_img(th1, th2, path, opt):
    img = cv2.imread(path)
    # thresh = cv2.Canny(image, th1, th2)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # img = draw_min_rect_circle(image, contours, opt)
    # 改变格式成规定的框和高
    if img.shape[2] != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_0 = cv2.resize(img, (opt.w, opt.h))
    img1 = np.zeros([3, opt.w, opt.h])  # 改变img1的时候不改变img
    img1[0, :, :] = img_0[:, :, 2]
    img1[1, :, :] = img_0[:, :, 1]
    img1[2, :, :] = img_0[:, :, 0]  # cv2读取的是bgr,转换成rgb就要做一下变通
    return img1


