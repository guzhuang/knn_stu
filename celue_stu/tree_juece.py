from math import log

def calcShannonEnt(dataSet):
    #计算数据集中实例总数
    numEtries = len(dataSet)
    #键值
    labelCount = {}
    #循环读取实例
    for featVec in dataSet:
        #取每个实例中的最后一列数值
        currentLabel = featVec[-1]
        #判断
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel] = 0
        #赋值
        labelCount[currentLabel] += 1
    shannonEnt = 0.0
    #计算香农熵
    for key in labelCount:
        prob = float(labelCount[key])/numEtries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

def createDataSet():
    dataSet = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing','fippers']
    return dataSet,labels

dataSet,labels = createDataSet()
#print(calcShannonEnt(dataSet))


def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            #列表的append方法把reducedFeatVec整个列表的所有元素和现有合起来，，一维数组还是一维数组
            #列表的append方法把reducedFeatVec整个列表作为一个元素,形成子列表，，，一维数组变为二维数组
            retDataSet.append(reducedFeatVec)
    return retDataSet

#myDat,labels = createDataSet()
print(splitDataSet(dataSet,0,1))

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    bestEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0;bestFeature = -1
    for i in range(numFeatures):
        featList = [Example[i] for Example in dataSet]
        uniqueVals = set(featList)
        newEntropy  = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = bestEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

print(chooseBestFeatureToSplit(dataSet))