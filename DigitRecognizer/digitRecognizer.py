"""
Created on 2016/09/06
@author:zhangqiang

"""

import csv
from numpy import *
import operator
import time


def loadTrainData():
    with open('data/train.csv','rb') as fp:
        lines =csv.reader(fp)
        l = []
        for line in lines:
            # print line
            l.append(line)#42001*785
    l.remove(l[0])
    l = array(l)
    labels = l[:,0]
    data = l[:,1:]
    return normalizing(toInt(data)),toInt(labels)

#the data in csv file is string type,
#we need to transform string to int type.
def toInt(array):
    array = mat(array)#m,n = shape(array) needs matrix as input
    m,n = shape(array)
    newArray = zeros((m,n))#important!!
    for i in xrange(m):
        for j in xrange(n):
            newArray[i,j] = int(array[i,j])
    return newArray

#the pixel value is in range 0~255,
#we need normalize nonzero value as 1.
def normalizing(array):
    array = mat(array)#m,n = shape(array) needs matrix as input
    m,n = shape(array)
    for i in xrange(m):
        for j in xrange(n):
            if array[i,j] != 0:
                array[i,j] = 1
    return array

# loadTrainData()

def loadTestData():
    with open('data/test.csv','rb') as fp:
        lines =csv.reader(fp)
        l = []
        for line in lines:
            # print line
            l.append(line)#28001*784
    l.remove(l[0])
    data = array(l)
    return normalizing(toInt(data))

def loadTestResult():
    with open('data/rf_benchmark.csv','rb') as fp:
        lines =csv.reader(fp)
        l = []
        for line in lines:
            # print line
            l.append(line)#28001*2
    l.remove(l[0])
    l = array(l)
    labels = l[:,1]
    return toInt(labels)

def knnClassifier(inX, dataSet, labels, k):
    """
    :param inX: one test sample
    :param dataSet: the train data
    :param labels: the train labels
    :param k: the value in range 0~20
    :return: the label of one test sample
    """
    inX=mat(inX)
    dataSet=mat(dataSet)
    labels=mat(labels)
    dataSetSize = dataSet.shape[0]
    # print inX,dataSet
    # print type(inX[0]),type(dataSet[0][0])
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = array(diffMat)**2#diffMat must be array type,not matrix type
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    # print sortedDistIndicies
    classCount={}
    for i in range(k):
        voteIlabel = labels[0,sortedDistIndicies[i]]#DEBUG:labels index losing 0
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def saveResult(result):
    with open('result.csv','wb') as fp:
        myWriter = csv.writer(fp)
        for i in result:
            tmp = []
            tmp.append(i)
            myWriter.writerow(tmp)


def digitRecognizer():
    t0 = time.clock()
    trainData,trainLabels = loadTrainData()
    testData = loadTestData()
    trueTestLabels = loadTestResult()
    t1 = time.clock()
    m = shape(testData)[0]
    result = []
    errorCount = 0
    for i in range(m):
        predTestLabel = knnClassifier(testData[i], trainData[0:20000],trainLabels[0:20000], 5)
        result.append(predTestLabel)
        print "the predict label is:%d, the true label is:%d"%(predTestLabel,trueTestLabels[0,i])
        if predTestLabel != trueTestLabels[0,i]:
            errorCount += 1
    t2 = time.clock()
    print "\nthe total error number of recognizer is: %d"%errorCount
    print "\nthe total error rate is: %f"%(errorCount/float(m))
    saveResult(result)
    t3 = time.clock()
    print "\nthe load data time is: %f"%(t1 - t0)
    print "\nthe classify time is: %f"%(t2 -t1)
    print "\nthe save time is:%f"%(t3 -t2)


digitRecognizer()

