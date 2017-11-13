#knn 算法
from numpy import *
import operator
import sys
import os

sys.path.append('D:\study\suanfa\knn_stu')


#创建数据集
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels
#输出数据集
group,labels = createDataSet()
#print(group,labels)

#k临近算法
## inX 用来分类的输入向量
## dataSet 训练样本集
## labels 标签向量
def classify0(inX,dataSet,labels,k):
    #计算巡逻组的行数
    dataSetSize = dataSet.shape[0]
    #tile 作用：把inX向量重复dataSetSize 次，相当于计算xA - xB
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    #（xA - xB）**2
    sqDiffMat = diffMat**2
    #求和，矩阵每一行向量相加
    sqDistance = sqDiffMat.sum(axis = 1)
    #开方，得出欧式距离
    distance = sqDistance ** 0.5
    #排序，升序排列
    sortedDistIndicies=distance.argsort()
    classCount={}
    #print(range(k))
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #items在python2中是iteritems。
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]


# img2Vector 图像转换为向量
def img2Vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

#testVect = img2Vector('img_data/testDigits/0_12.txt') 
#print(testVect)

def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('img_data/testDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2Vector('img_data/testDigits/%s' % fileNameStr)
    testFileList = os.listdir('img_data/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2Vector('img_data/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
       # print("the classifier come back with %d,the real result is: %d" % (classify0,classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\n the totalNum of errors is:%d" % errorCount)
    print("\n the total error rates ； %f" % (errorCount/float(mTest)))

handwritingClassTest()

