#!/usr/bin/python
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
mistakeArrayList = []
for i in range(10):
    inputFile = "ocr_fold" + str(i)+ "_sm_train.txt"
    lines = [line.rstrip('\n') for line in open(inputFile)]
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
    for i in range(len(data)):
        array = []
        for letter in data[i]:
            array.append(ord(letter) - 48)
        inputList.append(array)

    w = np.zeros(128)
    mistake = 0
    iterationArray = []
    for index in range(50):
        for j in range(len(data)):
            X = np.array(inputList[j])
            dotProductValueXW = np.dot(X,w)
            if dotProductValueXW > -1:
                sign = 1
            else:
                sign = -1
            if sign != output[j]:
                mistake += 1
                w = w + np.dot(X, output[j])
        mistakeArrayList.append(mistake)
        iterationArray.append(index)
        mistake = 0

avgmistakelist = []
for i in range(50):
    sumofMistake = 0
    for j in range(10):
        sumofMistake = sumofMistake + mistakeArrayList[j*50+i]
    avgmistakelist.append(sumofMistake/10)

plt.plot(iterationArray, avgmistakelist, marker='o')
plt.xlabel('Number of Iterations')
plt.ylabel('Number of Mistakes')
plt.show()