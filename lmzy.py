#!/usr/bin/python
# -*- coding: UTF-8 -*-

import cv2
import numpy

def getImage():
    img = cv2.imread("/home/xhq/data/lmzy/qm_201836_17.bmp")
    img=img[230:270].transpose(1,0,2)[20:112].transpose(1,0,2)
    #img=cv2.resize(img,(img_shape[1]/2,img_shape[0]/2),interpolation=cv2.INTER_CUBIC)
    return img

def fenci(img,thresh):


    _1, contours, _2 = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 根据轮廓列表，循环在原始图像上绘制矩形边界

    count_fenci=0
    for i in range(len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        if w*h>50:
            count_fenci+=1
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    print u"粉刺：",count_fenci
    return img

def imgShowSave(img,savePath):
    cv2.namedWindow("Image")
    cv2.imshow("Image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #cv2.imwrite(savePath,img)
    
if __name__=='__main__':
    #cv2.cvCreateGLCM()
    img=getImage()
    img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hsv=cv2.split(img_hsv)

    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (T, thresh) = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)#阈值化处理，阈值为：155

    img=fenci(img,hsv[1])

    laplacian=cv2.Laplacian(gray,cv2.CV_64F)

    # 参数 1,0 为只在 x 方向求一阶导数,最大可以求 2 阶导数。
    sobelx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
    # 参数 0,1 为只在 y 方向求一阶导数,最大可以求 2 阶导数。
    sobely=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)
    bgr=cv2.split(img)
    cv2.imshow('ss',img)
    cv2.waitKey(0)
    for i in range(1):
    	cv2.imshow('ss',hsv[i])
    	cv2.waitKey(0)


  #  imgShowSave(gray,'out.bmp')