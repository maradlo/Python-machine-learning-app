"""
1. metoda - cisto offline.. pridat percenta uspesnosti
offline/offline
"""

# Imports
import cv2 as cv
import numpy as np
import os
from tensorflow.keras import datasets, models, layers
import matplotlib.pyplot as plt

# import tensorflow
# import matplotlib

path = 'testing_images'
# Using ORB because it is freeware, swift for example is not free
orb = cv.ORB_create(nfeatures=1000)
# Importing images ###
images = []
classNames = []
testingImagesList = os.listdir(path)

# thres = 12

print('Total Classes Detected', len(testingImagesList))

# cl = class; imgCur = image current
for cl in testingImagesList:
    imgCur = cv.imread(f'{path}/{cl}', 0)
    images.append(imgCur)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)
# /Importing images ###

def findDescriptor(images):
    #descriptor list
    desList = []
    for image in images:
        keyPoints, descriptor = orb.detectAndCompute(image, None)
        desList.append(descriptor)
    return desList

def findID(image, desList):
    keyPoints2, descriptor2 = orb.detectAndCompute(image, None)
    brute_force_matcher = cv.BFMatcher()
    matchList = []
    finalValue = -1
    distanceNumber = 0.80
    thres = 8
    try:
        for des in desList:
            matches = brute_force_matcher.knnMatch(des, descriptor2, k=2)
            good_match = []
            for m, n in matches:
                if m.distance < distanceNumber * n.distance:
                    good_match.append([m])
            matchList.append(len(good_match))
    except:
        pass
    # print(matchList)
    if len(matchList) != 0:
        if max(matchList) > thres:
            finalValue = matchList.index(max(matchList))
    return finalValue

desList = findDescriptor(images)
print(len(desList))

cap = cv.VideoCapture(0)

while True:
    success, image2 = cap.read()
    imageOriginal = image2.copy()
    image2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
    # show how many matches it found..
    id = findID(image2, desList)
    if id != -1:
        cv.putText(imageOriginal, classNames[id], (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2)

    cv.imshow('image2', imageOriginal)
    if cv.waitKey(1) == ord('q'):
        cap.release()
        cv.destroyAllWindows()
        break

# x iteracie
# y chybovost

# vykreslovat len body..
x = np.arange(1, 11) 
y = x * x

plt.title("Line graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.plot(x, y, color ="red")
plt.show()
    

