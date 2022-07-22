# Imports
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image1 = cv.imread('testing_images/Rotor 1736.jpg', 0)
image2 = cv.imread('testing_images/9.jpg', 0)

# Using ORB because it is free, swift for example is not free
orb = cv.ORB_create(nfeatures=1000)

key_points1, descriptor1 = orb.detectAndCompute(image1, None)
key_points2, descriptor2 = orb.detectAndCompute(image2, None)

brute_force_matcher = cv.BFMatcher()
matches = brute_force_matcher.knnMatch(descriptor1, descriptor2, k=2)

good_match = []
x_points = 0
y_points = 0
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_match.append([m])
        x_points = x_points + 1
print(len(good_match))

image3 = cv.drawMatchesKnn(image1, key_points1, image2, key_points2, good_match, None, matchColor=(0, 255, 0), matchesMask=None,
                              singlePointColor=(255, 0, 0), flags=2)

imgDimension =(150, 150)
font = cv.FONT_HERSHEY_SIMPLEX
fontScale = 4
color = (255, 0, 0)
thickness = 2

finalImage = cv.putText(image3, 'Zhodne body ' +str(len(good_match)), imgDimension, font, fontScale, color, thickness, cv.LINE_AA )

# # data to be plotted
# x = np.arange(0,1000)
# y = np.arange(0,x_points)
 
# # plotting
# plt.style.use('seaborn-pastel')
# plt.title("Graf zhodných bodov")
# plt.xlabel("Iterácie")
# plt.ylabel("Zhodné body")
# plt.plot(x, y, color ="red")
# plt.show()

cv.imshow('Výsledok', finalImage)
cv.waitKey(0)