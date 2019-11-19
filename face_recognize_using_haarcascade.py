# import the necessary packages
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
from PIL import Image

cwd = os.getcwd()
pathSeparator = "/"
if "\\" in cwd:
    pathSeparator = "\\"

cascPath = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
cascEyePath = cv2.data.haarcascades + 'haarcascade_eye.xml'

# load the image, resize it to have a width of 600 pixels (while
# maintaining the aspect ratio), and then grab the image dimensions
# image = cv2.imread("C:\\Users\\Acer\\Documents\\GitHub\\hello\\images\\")

# (h, w) = image.shape[:2]

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch("nn4.small2.v1.t7")


# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(cwd + pathSeparator +"output" + pathSeparator + "recognizer.pickle", "rb").read())
le = pickle.loads(open(cwd + pathSeparator +"output" + pathSeparator + "le.pickle", "rb").read())


face_cascade = cv2.CascadeClassifier(cascPath)
eye_cascade = cv2.CascadeClassifier(cascEyePath)


video_capture = cv2.VideoCapture(0)

lastX=0
lastY=0
distance = 10
while True:

    ret, frame = video_capture.read(0)

    frame = imutils.resize(frame, width=600)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100,100))
 
    for (x, y, w, h) in faces:
            if(w < 100 and h < 100): continue

            face = frame[y:y+h, x:x+w]
            roi_gray = gray[y:y+h, x:x+w]
            # eyes = eye_cascade.detectMultiScale(roi_gray)
            # for (ex,ey,ew,eh) in eyes:
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                (0, 0, 0), swapRB=True, crop=False)

            embedder.setInput(faceBlob)
            vec = embedder.forward()
            
            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            if(proba < 0.3):
                continue
            name = le.classes_[j]
            if(name == "unknow"):
                print(name)
                continue
                
            text = "{}: {:.2f}%".format(name, proba * 100)
            
            cv2.rectangle(frame, (x,y), (x + w, y + h,), (255, 0, 0, 0), 2)
            cv2.putText(frame, text, (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imshow('img', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows

###############################################################################

# cascPath = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# recognizeResult = ''


# print("[INFO] loading face recognizer...")
# embedder = cv2.dnn.readNetFromTorch("nn4.small2.v1.t7")


# recognizer = pickle.loads(open("C:\\Users\\Acer\\Documents\\GitHub\\hello\\output\\recognizer.pickle", "rb").read())
# le = pickle.loads(open("C:\\Users\\Acer\\Documents\\GitHub\\hello\\output\\le.pickle", "rb").read())



# face_cascade = cv2.CascadeClassifier(cascPath)

# def recognize(frame):
#     result = None

#     global recognizeResult

#     frame = imutils.resize(frame, width=600)
    
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     faces = face_cascade.detectMultiScale(gray, 1.3, 1)
 
#     for (x, y, w, h) in faces:

#         face = frame[y:y+h, x:x+w]

#         faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
#             (0, 0, 0), swapRB=True, crop=False)

#         embedder.setInput(faceBlob)
#         vec = embedder.forward()
        
#         # perform classification to recognize the face
#         preds = recognizer.predict_proba(vec)[0]
#         j = np.argmax(preds)
#         proba = preds[j]
#         if(proba < 0.3):
#             continue
#         name = le.classes_[j]
#         if(name == "unknow"):
#             print(name)
#             continue
#         recognizeResult = name
#         text = "{}: {:.2f}%".format(name, proba * 100)
        
#         cv2.rectangle(frame, (x,y), (x + w, y + h,), (255, 0, 0, 0), 2)
#         cv2.putText(frame, text, (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
#         result = frame
#     return result


###############################################################################



