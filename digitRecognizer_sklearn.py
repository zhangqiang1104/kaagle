"""
Created on 2016/09/06
@author:zhangqiang

"""

import csv
from numpy import *
import time


def loadTrainData():
    with open('data/train.csv','rb') as fp:
        lines =csv.reader(fp)
        l = []; i = 0
        for line in lines:
            # print line
            i += 1
            if i < 20000:
                l.append(line)#42001*785
            else: break
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

def saveResult(result,filename):
    with open(filename,'wb') as fp:
        myWriter = csv.writer(fp)
        for i in result:
            tmp = []
            tmp.append(i)
            myWriter.writerow(tmp)


from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

def knnClf(trainData,trainLabel,testData):
    knnClf = KNeighborsClassifier()
    t1 = time.clock()
    knnClf.fit(trainData,ravel(trainLabel))
    predTestLabels = knnClf.predict(testData)

    m = shape(testData)[0]
    errorCount = 0
    for i in range(m):
        if predTestLabels[i] != trueTestLabels[0,i]:
            errorCount += 1
    t2 = time.clock()
    print "\nthe total error number of recognizer is: %d"%errorCount
    print "the total error rate is: %f"%(errorCount/float(m))
    saveResult(predTestLabels,'result_sklearn_knn.csv')
    print "the knn time is: %f"%(t2 -t1)

def svmClf(trainData,trainLabel,testData):
    svmClf = svm.SVC()
    t1 = time.clock()
    svmClf.fit(trainData,ravel(trainLabel))
    predTestLabels = svmClf.predict(testData)

    m = shape(testData)[0]
    errorCount = 0
    for i in range(m):
        if predTestLabels[i] != trueTestLabels[0,i]:
            errorCount += 1
    t2 = time.clock()
    print "\nthe total error number of recognizer is: %d"%errorCount
    print "the total error rate is: %f"%(errorCount/float(m))
    saveResult(predTestLabels,'result_sklearn_svm.csv')
    print "the svm time is: %f"%(t2 -t1)

def bayesGausClf(trainData,trainLabel,testData):
    bayesGausClf = GaussianNB()
    t1 = time.clock()
    bayesGausClf.fit(trainData,ravel(trainLabel))
    predTestLabels = bayesGausClf.predict(testData)

    m = shape(testData)[0]
    errorCount = 0
    for i in range(m):
        if predTestLabels[i] != trueTestLabels[0,i]:
            errorCount += 1
    t2 = time.clock()
    print "\nthe total error number of recognizer is: %d"%errorCount
    print "the total error rate is: %f"%(errorCount/float(m))
    saveResult(predTestLabels,'result_sklearn_bayesGaus.csv')
    print "the bayesGaus time is: %f"%(t2 -t1)

def bayesMultiClf(trainData,trainLabel,testData):
    bayesMultiClf = MultinomialNB()
    t1 = time.clock()
    bayesMultiClf.fit(trainData,ravel(trainLabel))
    predTestLabels = bayesMultiClf.predict(testData)

    m = shape(testData)[0]
    errorCount = 0
    for i in range(m):
        if predTestLabels[i] != trueTestLabels[0,i]:
            errorCount += 1
    t2 = time.clock()
    print "\nthe total error number of recognizer is: %d"%errorCount
    print "the total error rate is: %f"%(errorCount/float(m))
    saveResult(predTestLabels,'result_sklearn_bayesMulti.csv')
    print "the bayesMulti time is: %f"%(t2 -t1)

trainData,trainLabel = loadTrainData()
testData = loadTestData()
trueTestLabels = loadTestResult()

svmClf(trainData,trainLabel,testData)
bayesGausClf(trainData,trainLabel,testData)
bayesMultiClf(trainData,trainLabel,testData)
knnClf(trainData,trainLabel,testData)
