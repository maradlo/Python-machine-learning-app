import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, models, layers

#pridat % hodnotenie loss, accuracy = model.evaluate(testing_images, testing_labels)
#zamerat sa na cisla a stoppoint

img1 = cv.imread('../training_images/1634.jpeg', 0)
img2 = cv.imread('../testing_images/1634.jpeg', 0)

#ORB is fast and free
orb = cv.ORB_create(nfeatures=1000)

#kp - key points ; ds - descriptors
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

#imgKp1 = cv.drawKeypoints(img1,kp1,None)
#imgKp2 = cv.drawKeypoints(img2,kp2,None)

#brutforce matcher

bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

good = []
for m,n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

#cv.imshow('Kp1', imgKp1)
#cv.imshow('Kp2', imgKp2)
#cv.imshow('img1', img1)
#cv.imshow('img2', img2)
cv.imshow('img3', img3)

cv.waitKey(0)




# img = cv.imread('1630.jpeg', -1)
# cv.imshow('image', img)
# k = cv.waitKey(0)

# if k == 27:
#     cv.destroyAllWindows()
# elif k == ord('s'):
#     cv.imwrite('1630_copy.png', img)
#     cv.destroyAllWindows()



# # load_data() function loads training and testing data in this format - img is array of pixels, labels are labels
# (training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

# training_images, testing_images = training_images / 255, testing_images / 255

# class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
# #class_names = ['1630', '1634', '1636', '1720', '1736']

# # #vizualizacia 16 obrazkov
# # for i in range(16):
# #     plt.subplot(4,4,i+1) #4x4 mriezka, kazda dalsia iteracia vyberie dalsi stvorec
# #     #ziadne koordinacie 
# #     plt.xticks([])
# #     plt.yticks([])
# #     plt.imshow(training_images[i], cmap=plt.cm.binary)
# #     plt.xlabel(class_names[training_labels[i][0]]) #ziskame label daneho obrazku(index) a potom ju posielame class_names ako index 

# # plt.show()

# #reducing amount of images that we are feeding neural networks with
# #helps with performance on slower machines
# training_images = training_images[:20000]
# training_labels = training_labels[:20000]
# testing_images = testing_images[:4000]
# testing_labels = testing_labels[:4000]

# # #Build of neural network
# # #Conv2D is filtering differences in pictures ==> horse has long legs, plane has wings

# # model = models.Sequential() #Definition of neural network
# # model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3))) #input layer 
# # model.add(layers.MaxPooling2D((2,2))) #Simplifies result and reduced it to esencial informations
# # model.add(layers.Conv2D(64, (3,3), activation='relu'))
# # model.add(layers.MaxPooling2D((2,2)))
# # model.add(layers.Conv2D(64, (3,3), activation='relu'))
# # model.add(layers.Flatten())
# # model.add(layers.Dense(64, activation='relu'))
# # model.add(layers.Dense(10, activation='softmax'))

# # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images,testing_labels))

# # loss, accuracy = model.evaluate(testing_images, testing_labels)

# # print(f"Loss: {loss}")
# # print(f"Accuracy: {accuracy}")

# # model.save('image_classifier.model')


# model = models.load_model('image_classifier.model')

# img = cv.imread('images/plane.jpg')
# img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# plt.imshow(img, cmap=plt.cm.binary)

# prediction = model.predict(np.array([img]) / 255)
# index = np.argmax(prediction)
# print(f'Prediction is {class_names[index]}')

# plt.show()