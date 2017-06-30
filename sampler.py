import os
import threading
import random
import time

import numpy as np
import cv2
import tensorflow as tf

import classfier

dataList=[]
dataLabelList=[]
isStarted=False
isInited=False

maxCachedData=200
cachedImageList=[]
cachedIndexList=[]
lock=threading.Lock()

imageDir=["face","body","background"]

def initSampler(path):
    global isInited,isStarted
    if isStarted!=False:return
    isStarted=True

    thread=threading.Thread(target=threadFun,args=[path])
    thread.setDaemon(True)
    thread.start()

def threadFun(path):
    global isInited,isStarted

    if loadImageData(path)!=True:
        isStarted=False
        return
    isInited=True
    fillCache()

def loadImageData(path):
    global dataList,dataLabelList

    dataLabelList=[]
    dataList=[]

    for dir in imageDir:
        dirPath=os.path.join(path,dir)
        if os.path.isdir(dirPath)!=True:continue

        dataLabelList.append(dir)
        tempList=[]
        for image in os.listdir(dirPath):
            imagePath=os.path.join(dirPath,image)
            tempList.append(cv2.resize(cv2.imread(imagePath),classfier.IMAGE_SIZE))
        dataList.append(tempList)

    return True

import tensorflow as tf
input_image=tf.placeholder(tf.float32,[classfier.IMAGE_SIZE[0],classfier.IMAGE_SIZE[1],3])
output_image=tf.image.per_image_standardization(input_image)
session=tf.Session()
def preProcessImage(image):
    return session.run([output_image],feed_dict={input_image:image})[0]

def fillCache():
    global dataList
    global lock,cachedImageList,cachedIndexList
    while True:
        if len(cachedImageList)>=maxCachedData:
            time.sleep(0.01)
            continue

        indexList=[]
        imageList=[]
        for i in range(maxCachedData):
            index=random.randint(0,len(dataList)-1)
            image=dataList[index][random.randint(0,len(dataList[index])-1)]
            image=transfromImage(image,0.2,0.2,0.2).astype(np.float32)
            image=preProcessImage(image)

            indexList.append(index)
            imageList.append(image)

        lock.acquire()
        cachedIndexList+=indexList
        cachedImageList+=imageList
        lock.release()


def transfromImage(image,fRandomFactor,hRandomFactor, vRandomFactor):
    shape=image.shape[0:2]

    fromPoint=[[0,0],[shape[0]-1,0],[0,shape[1]-1]]

    moveScale = shape[0]
    hMoveLength = moveScale * random.uniform(-hRandomFactor, hRandomFactor)
    vMoveLength = moveScale * random.uniform(-vRandomFactor, vRandomFactor)

    transformScale=shape[0]/4
    toPoint = np.array(fromPoint, np.float32)
    for i in range(3):
        toPoint[i][0]+=transformScale*random.uniform(-fRandomFactor,fRandomFactor)+hMoveLength
        toPoint[i][1] += transformScale * random.uniform(-fRandomFactor, fRandomFactor)+vMoveLength

    transfromMat=cv2.getAffineTransform(np.array(fromPoint,np.float32),toPoint)
    return cv2.warpAffine(image,transfromMat,shape,flags=cv2.INTER_LINEAR)


def rotateImage(image,randomFactor):
    shape=image.shape[0:2]
    angle=random.uniform(-randomFactor,randomFactor)
    rotateMat=cv2.getRotationMatrix2D((shape[0]/2,shape[1]/2),angle,1)
    return cv2.warpAffine(image,rotateMat,shape)

def check():
    if isStarted==True and isInited==False:
        while True:
            if isInited==False:
                time.sleep(5)
            else:
                break
    if isInited!=True:
        raise Exception("sampler is  not inited")
    if len(dataLabelList)!=len(dataList):
        raise Exception("sampler data is wrong")

def getBatch(num):
    global maxCachedData,lock,cachedImageList,cachedIndexList

    check()
    if num*2>maxCachedData:
        maxCachedData=2*num

    while True:
        if len(cachedImageList)<=num:
            print("sleep to get batch")
            time.sleep(0.01)

        lock.acquire()
        imageList=cachedImageList[0:num]
        cachedImageList=cachedImageList[num:len(cachedImageList)]
        indexList=cachedIndexList[0:num]
        cachedIndexList=cachedIndexList[num:len(cachedIndexList)]
        lock.release()

        return np.array(imageList),np.array(indexList)


def getLabel(index):
    if index<0 or index>=len(imageDir):
        raise Exception("invalid index")
    return imageDir[index]