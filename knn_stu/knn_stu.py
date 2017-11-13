#knn 算法

from numpy import *
import operator
import sys

sys.path.append('D:\study\suanfa\knn_stu')
#创建数据集
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels
#输出数据集
group,labels = createDataSet()
print(group,labels)

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
    print(range(k))
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #items在python2中是iteritems。
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]

#sortedClassCount[0][0] =
print(classify0([1,0],group,labels,3))
#print(sortedClassCount[0][0]) 

def file2matrix(filename):
    #打开文件
    fr = open(filename)
    #读取文件全部行
    arrayOfLine = fr.readlines()
    #获取文件行数
    numberOfLines = len(arrayOfLine)
    #创建numpy全零矩阵
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOfLine:
        #strip()代表删掉空格\r\t\n等
        line = line.strip()
        #切片
        listFormLine = line.split('\t')
        returnMat[index,:] = listFormLine[0:3]
        classLabelVector.append(int(listFormLine[-1]))
        index += 1
    return returnMat,classLabelVector

datingDataMat,datingLabels = file2matrix('data/datingTestSet2.txt')
#print(datingDataMat,datingLabels[0:20] )


#归一化处理
#newValue = （oldValue - min）/max
def autoNormal(dataSet):
    minValues = dataSet.min()
    maxValues = dataSet.max()
    ranges = maxValues - minValues
    normalDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    #矩阵单列-min值
    normalDataSet = dataSet - tile(minValues,(m,1))
    #矩阵单列/max值
    normalDataSet = normalDataSet/tile(maxValues,(m,1))
    return normalDataSet,ranges,minValues

normalDataSet,ranges,minValues = autoNormal(datingDataMat)
print(normalDataSet,ranges,minValues)

#出预测
def classifyPerson():
    resultLists = ['not at all','in some doses','in large doses']
    #PYTHON 3 不支持raw_input,直接使用input
    percentTats = 2334 #float(input("花费在录像上的时间数："))
    ffMiles = 45678   #float(input("每年的飞行公里数："))
    iceCream = 90 #float(input("消耗冰淇淋数："))
    datingDataMat,datingLabels = file2matrix('data/datingTestSet2.txt')
    normalDataSet,ranges,minValues = autoNormal(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minValues)/ranges,normalDataSet,datingLabels,3)
    print("你对这个人的喜欢程度是：",resultLists[classifierResult - 1])

classifyPerson()
