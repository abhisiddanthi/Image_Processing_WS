import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

with_mask = np.load('with_mask.npy')
without_mask = np.load('without_mask.npy')

with_mask = with_mask.reshape(200, 50 * 50 * 3)
without_mask = without_mask.reshape(200, 50 * 50 * 3)

X = np.r_[with_mask, without_mask]
labels = np.zeros(X.shape[0])
labels[200:] = 1.0  

pca = PCA(n_components=3)
x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size = 0.25)
x_train = pca.fit_transform(x_train)
svm = SVC()
svm.fit(x_train, y_train)
x_test = pca.transform(x_test)
y_pred = svm.predict(x_test)

names = {0 : 'Mask', 1 : 'No Mask'}


cascPath = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
capture = cv2.VideoCapture(0)

while True:
    flag, img = capture.read()
    faces = faceCascade.detectMultiScale(img)   
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h),(255,0,255), 4)
        face = img[y:y+h, x:x+w, :]
        face = cv2.resize(face, (50,50))
        face = face.reshape(1, -1)
        face = pca.transform(face)
        pred = svm.predict(face)[0]
        n = names[int(pred)]
        print(n)
    cv2.imshow('Video', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    


capture.release()
cv2.destroyAllWindows()     