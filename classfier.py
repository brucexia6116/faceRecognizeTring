
IMAGE_SIZE=(128,128)

import threading

import tensorflow as tf
import numpy as np
import cv2

import sampler

def addConvLayer(input,shape,name):
    with tf.name_scope(name) as scope:
        weight=tf.Variable(tf.truncated_normal(shape,stddev=0.1),name="weight")
        bias=tf.Variable(tf.zeros([shape[3]]),name="bias")
        output=tf.nn.relu(tf.nn.conv2d(input,weight,[1,1,1,1],padding="SAME")+bias,name="output")
    weightLoss=tf.multiply(tf.nn.l2_loss(weight),0.01)
    tf.add_to_collection("loss",weightLoss)
    return output

def addFullLayer(input,shape,name):
    with tf.name_scope(name) as scope:
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name="weight")
        bias = tf.Variable(tf.zeros([shape[1]]), name="bias")
        output=tf.nn.relu(tf.matmul(input,weight)+bias,name="output")
    weightLoss = tf.multiply(tf.nn.l2_loss(weight), 0.01)
    tf.add_to_collection("loss",weightLoss)
    return output


input_image=tf.placeholder(tf.float32,[IMAGE_SIZE[0],IMAGE_SIZE[1],3])
output_image=tf.image.per_image_standardization(input_image)

inputImage=tf.placeholder(tf.float32,[None,IMAGE_SIZE[0],IMAGE_SIZE[1],3])
inputLabel=tf.placeholder(tf.int32,[None])

conv1=addConvLayer(inputImage,[7,7,3,8],"conv1")
pool1=tf.nn.max_pool(conv1,[1,4,4,1],[1,4,4,1],padding="SAME")
conv2=addConvLayer(pool1,[5,5,8,16],"conv2")
pool2=tf.nn.max_pool(conv2,[1,4,4,1],[1,4,4,1],padding="SAME")
conv3=addConvLayer(pool2,[5,5,16,32],"conv3")
pool3=tf.nn.max_pool(conv3,[1,4,4,1],[1,4,4,1],padding="SAME")

featureLength=int(IMAGE_SIZE[0]*IMAGE_SIZE[1]/128)
print("feature length="+str(featureLength))

reShape=tf.reshape(pool3,[-1,featureLength])
fc1=addFullLayer(reShape,[featureLength,32],"fc1")
fc2=addFullLayer(fc1,[32,3],"fc2")

logit=tf.nn.softmax(fc2)
outIndex=tf.argmax(fc2,axis=1)

cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=fc2,labels=inputLabel,name="cross_entropy")
tf.add_to_collection("loss",tf.reduce_mean(cross_entropy))
train_op=tf.train.AdamOptimizer().minimize(tf.add_n(tf.get_collection("loss")))
accurate=tf.reduce_sum(tf.cast(tf.equal(tf.cast(inputLabel,tf.int64),tf.argmax(fc2,1)),tf.float32))

initer=tf.global_variables_initializer()
saver=tf.train.Saver()

#######################################################################
BATCH=50
message=""
isStart=False
path=None

def startTrain():
    global isStart,path
    if isStart==True:
        return
    isStart=True
    thread=threading.Thread(target=threadFun,args=[tf.get_default_graph()])
    thread.setDaemon(True)
    thread.start()

def setSavePath(savePath):
    global path
    path=savePath

def threadFun(graph):
    global message,isStart,path

    sess=tf.Session(graph=graph)
    sess.run([initer])

    iterCount = np.zeros([200], np.float32)
    accurateCount = np.zeros([200], np.float32)
    curIter=-1
    while True:
        curIter+=1
        trainData=sampler.getBatch(BATCH)
        retAcc=sess.run([accurate,train_op],feed_dict={inputImage:trainData[0],inputLabel:trainData[1]})[0]

        iterCount[curIter % 200] = 50
        accurateCount[curIter % 200] = retAcc

        message="iter:"+str(curIter)+"  cur:"+str(int(retAcc))\
                +"  accurate:"+str(np.sum(accurateCount)/np.sum(iterCount))

        if path!=None:
            saver.save(sess,path)
            sess.close()
            isStart=False
            path = None
            message=""
            return

#######################################################################

classfySess=None

def loadModule(path):
    global classfySess
    if classfySess!=None:
        classfySess.close()
    classfySess=tf.Session()
    saver.restore(sess=classfySess,save_path=path)

def recognizeImage(image):
    if classfySess==None:
        raise Exception("classfy session is not inited")

    image=cv2.resize(image,IMAGE_SIZE).astype(np.float32)
    image=classfySess.run([output_image],feed_dict={input_image:image})[0]

    probability=classfySess.run([logit],feed_dict={inputImage:np.array([image])})[0][0]

    returnDic={}
    returnDic[sampler.getLabel(0)]=probability[0]
    returnDic[sampler.getLabel(1)]=probability[1]
    returnDic[sampler.getLabel(2)]=probability[2]

    return returnDic