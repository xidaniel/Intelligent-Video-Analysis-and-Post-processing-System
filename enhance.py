import cv2
import numpy as np

def buffing(img):
    dst = cv2.bilateralFilter(img, 15, 35, 35)
    return dst

def brightness(img, value=20):
    height = img.shape[0]
    width = img.shape[1]
    dst = np.zeros((height, width, 3), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            (b, g, r) = img[i, j]
            bb = int(b) + value
            gg = int(g) + value
            rr = int(r) + value
            if bb > 255:
                bb = 255
            if gg > 255:
                gg = 255
            if rr > 255:
                rr = 255
            dst[i, j] = (bb, gg, rr)

    return dst


def adaptive_threshold(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 把输入图像灰度化
    # 自适应阈值化能够根据图像不同区域亮度分布，改变阈值
    binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)
    return binary

def global_threshold(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  #把输入图像灰度化
    #自适应阈值化能够根据图像不同区域亮度分布，改变阈值
    ret, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    return binary

def mean_threshold(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 把输入图像灰度化
    mean = np.mean(img)
    print("mean:", mean)
    print(np.median(img))
    ret, binary = cv2.threshold(img, mean, 255, cv2.THRESH_BINARY)
    return binary

def go():
    img = cv2.imread('images/29.png')
    dst = mean_threshold(img)
    #edged = cv2.Canny(dst, 50, 200, True)
    #cv2.imshow('ori', img)
    #cv2.imshow('dst', dst)
    #cv2.imshow('edge', edged)
    #cv2.waitKey()

