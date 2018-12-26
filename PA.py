#!/usr/bin/python
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
accuraceArrayListTest = []
accuraceArrayList = []
for i in range(10):
    inputFile = "ocr_fold" + str(i)+ "_sm_train.txt"
    inputFileTest = "ocr_fold" + str(i) + "_sm_test.txt"
    lines = [line.rstrip('\n') for line in open(inputFile)]
    linesTest = [line.rstrip('\n') for line in open(inputFileTest)]

    data = []
    output = []
    for index in range(len(lines)):
        strings = lines[index].split()
        if strings:
            garbageData = strings[1]
            newData = garbageData.replace("im","")
            data.append(newData)
            for letter in strings[2]:
                if letter in ('a', 'e','i', 'o', 'u'):
                    output.append(1)
                else:
                    output.append(-1)
    inputList = []
    for index in range(len(data)):
        array = []
        for letter in data[index]:
            array.append(ord(letter) - 48)
        inputList.append(array)

    dataTest = []
    outputTest = []
    for index in range(len(linesTest)):
        strings = linesTest[index].split()
        if strings:
            garbageData = strings[1]
            newData = garbageData.replace("im","")
            dataTest.append(newData)
            for letter in strings[2]:
                if letter in ('a', 'e','i', 'o', 'u'):
                    outputTest.append(1)
                else:
                    outputTest.append(-1)
    inputListTest = []
    for index in range(len(dataTest)):
        array = []
        for letter in dataTest[index]:
            array.append(ord(letter) - 48)
        inputListTest.append(array)

    w = np.zeros(128)
    mistake = 0
    mistakeTest = 0
    iterationArray = []
    for trainingIteration in range(50):
        iterationArray.append(trainingIteration)
        for j in range(len(data)):
            x = np.array(inputList[j])
            dotProductValue = np.dot(x,w)
            sign = dotProductValue * output[j]
            if sign < 1:
                norm = np.linalg.norm(x, 2)
                T = (1 - (output[j]*dotProductValue))/(norm*norm)
                w = w + (T * x * output[j])
        for j in range(len(data)):
            x = np.array(inputList[j])
            dotProductValue = np.dot(x, w)
            sign = dotProductValue * output[j]
            if sign < 1:
                mistake += 1

        numberOfInput = len(data)
        accuracy = 100 - ((mistake/numberOfInput)*100)
        accuraceArrayList.append(accuracy)
        mistake = 0

        for jTest in range(len(dataTest)):
            x = np.array(inputListTest[jTest])
            dotProductValue = np.dot(x,w)
            sign = dotProductValue * outputTest[jTest]
            if sign < 1:
                mistakeTest += 1
        numberOfInput = len(dataTest)
        accuracy = 100 - ((mistakeTest / numberOfInput) * 100)
        accuraceArrayListTest.append(accuracy)
        mistakeTest = 0

avgAccuracylist = []
avgAccuracylistTest = []
for i in range(50):
    sumofAccuracy = 0
    sumofAccuracyTest = 0
    for j in range(10):
        sumofAccuracy = sumofAccuracy + accuraceArrayList[j*50+i]
        sumofAccuracyTest = sumofAccuracyTest + accuraceArrayListTest[j * 50 + i]
    avgAccuracylist.append(sumofAccuracy/10)
    avgAccuracylistTest.append(sumofAccuracyTest / 10)

plt.plot(iterationArray, avgAccuracylist, marker='o', label='Training Data')
plt.plot(iterationArray, avgAccuracylistTest, marker='o', linestyle='--', color='r', label='Testing Data')
plt.xlabel('Number of Iterations')
plt.ylabel('Accuracy(%)')
plt.legend()
plt.show()