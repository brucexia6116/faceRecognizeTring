import classfier

MIN_SREACH_SIZE=20

def getFaceRegion(image):
    retDic,retRect=recognizeFace(image,(0,image.shape[0],0,image.shape[1]))
    if retDic["background"] >= 1.0 / 3:
        return None
    return retRect[0]/image.shape[0],retRect[1]/image.shape[0],retRect[2]/image.shape[1],retRect[3]/image.shape[1]

'''
subRegionList=[(0,0.6,0,1),(0.4,1,0,1),(0,1,0,0.6),(0,1,0.4,1)]
def recognizeFace(image,rect):
    retDic=classfier.recognizeImage(image[rect[0]:rect[1],rect[2]:rect[3]])
    if retDic["background"]>=1.0/3:
        return retDic,rect

    width=rect[1]-rect[0]
    height=rect[3]-rect[2]
    if  width<=MIN_SREACH_SIZE or height<=MIN_SREACH_SIZE:
        return retDic,rect

    goodDic=retDic
    goodRect=rect

    for subRegion in subRegionList:
        subRect=[]
        subRect.append(rect[0]+int(subRegion[0]*width))
        subRect.append(rect[0] + int(subRegion[1] * width))
        subRect.append(rect[2] + int(subRegion[2] * height))
        subRect.append(rect[2] + int(subRegion[3] * height))

        subRetDic,subRetRect=recognizeFace(image,subRect)
        if subRetDic["face"]>goodDic["face"]:
            goodDic=subRetDic
            goodRect=subRetRect

    return goodDic,goodRect
'''

def getSizeReword(big,small):
    rate=((big[1]-big[0])*(big[3]-big[2]))/((small[1]-small[0])*(small[3]-small[2]))
    return rate/15

#'''
subRegionList=[(0,0.8,0,1),(0.2,1,0,1),(0,1,0,0.8),(0,1,0.2,1),(0.2,0.8,0,1),(0,1,0.2,0.8),(0.2,0.8,0.2,0.8)]
def recognizeFace(image,rect):
    retDic={"background":1,"face":0,"body":0}
    retRect=rect

    while True:
        width = retRect[1] - retRect[0]
        height = retRect[3] - retRect[2]
        if width <= MIN_SREACH_SIZE or height <= MIN_SREACH_SIZE:
            break

        goodDic=None
        goodRect=None
        for subRegion in subRegionList:
            subRect = []
            subRect.append(rect[0] + int(subRegion[0] * width))
            subRect.append(rect[0] + int(subRegion[1] * width))
            subRect.append(rect[2] + int(subRegion[2] * height))
            subRect.append(rect[2] + int(subRegion[3] * height))

            subRetDic=classfier.recognizeImage(image[subRect[0]:subRect[1],subRect[2]:subRect[3]])
            if goodDic==None or subRetDic["face"]>=goodDic["face"]:
                goodDic=subRetDic
                goodRect=subRect
        if (goodDic["face"]+getSizeReword(retRect,goodRect))<retDic["face"] or goodDic["background"]>1/3:
            break
        else:
            retDic=goodDic
            retRect=goodRect

    return retDic, retRect
#'''