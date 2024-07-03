import cv2
import os
import numpy as np

data = []

cascPath = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
capture = cv2.VideoCapture(0)


while True:
    img = cv2.imread('./test/nomask.png')
    faces = faceCascade.detectMultiScale(img)   
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h),(255,0,255), 4)
        face = img[y:y+h, x:x+w, :]
        face = cv2.resize(face, (50,50))
        print(len(data))
        if len(data) < 400:
            data.append(face)   
            
    cv2.imshow('Video', img)
    if cv2.waitKey(1) & 0xFF == 27 or len(data) >= 50:
        break
    
while True:
    img = cv2.imread('./test/guy.jpg')
    faces = faceCascade.detectMultiScale(img)   
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h),(255,0,255), 4)
        face = img[y:y+h, x:x+w, :]
        face = cv2.resize(face, (50,50))
        print(len(data))
        if len(data) >= 50:
            data.append(face)   
            
    cv2.imshow('Video', img)
    if cv2.waitKey(1) & 0xFF == 27 or len(data) >= 100:
        break
    
while True:
    img = cv2.imread('./test/guy3.jpg')
    faces = faceCascade.detectMultiScale(img)   
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h),(255,0,255), 4)
        face = img[y:y+h, x:x+w, :]
        face = cv2.resize(face, (50,50))
        print(len(data))
        if len(data) >= 100:
            data.append(face)   
            
    cv2.imshow('Video', img)
    if cv2.waitKey(1) & 0xFF == 27 or len(data) >= 150:
        break

while True:
    img = cv2.imread('./test/guy1.jpg')
    faces = faceCascade.detectMultiScale(img)   
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h),(255,0,255), 4)
        face = img[y:y+h, x:x+w, :]
        face = cv2.resize(face, (50,50))
        print(len(data))
        if len(data) >=150:
            data.append(face)   
            
    cv2.imshow('Video', img)
    if cv2.waitKey(1) & 0xFF == 27 or len(data) >= 200:
        break

np.save('without_mask.npy' ,data)
capture.release()
cv2.destroyAllWindows()     