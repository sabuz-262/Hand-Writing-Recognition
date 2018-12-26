from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
accuraceArrayListTest = []
accuraceArrayList = []
for i in range(10):
    print i
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
    iterationArray = []
    for numberOfIteration in range(50):
        iterationArray.append(numberOfIteration)
        w = np.zeros(128)
        u = np.zeros(128)
        c = 1
        mistakeTest = 0
        mistake = 0
        for index in range(numberOfIteration):
            for j in range(len(data)):
                X = np.array(inputList[j])
                dotProductValueXW = np.dot(X,w)
                sign = -1
                if dotProductValueXW > -1:
                    sign = 1
                elif dotProductValueXW < 0:
                    sign = -1
                if sign * output[j] <= 0:
                    w = w + np.dot(X, output[j])
                    u = u + output[j]*c*X
                c += 1
        w = w -(1/c)*u
        for jTrain in range(len(data)):
            X = np.array(inputList[jTrain])
            dotProductValueXW = np.dot(X, w)
            sign = -1
            if dotProductValueXW > -1:
                sign = 1
            elif dotProductValueXW < 0:
                sign = -1
            if sign * output[jTrain] <= 0:
                mistake += 1
        numOfInput = len(data)
        accuracy = (1 - (mistake/numOfInput)) * 100
        accuraceArrayList.append(accuracy)

        for jTest in range(len(dataTest)):
            X = np.array(inputListTest[jTest])
            dotProductValueXW = np.dot(X, w)
            sign = -1
            if dotProductValueXW > -1:
                sign = 1
            elif dotProductValueXW < 0:
                sign = -1
            if sign * outputTest[jTest] <= 0:
                mistakeTest += 1

        numOfInput = len(dataTest)
        accuracy = (1 - (mistakeTest / numOfInput)) * 100
        accuraceArrayListTest.append(accuracy)
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

plt.plot(iterationArray, avgAccuracylist, marker='o', label = 'Training Data')
plt.plot(iterationArray, avgAccuracylistTest, marker='o', linestyle='--', color='r',label = 'Testing Data')
plt.xlabel('Number of Iterations')
plt.ylabel('Accuracy(%)')
plt.legend()
plt.show()