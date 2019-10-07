import numpy as np
import cv2 as cv
from PIL import Image
import time

cascPath = cv.data.haarcascades + 'haarcascade_frontalface_default.xml'
system_path = "C:\\Users\\Acer\\Desktop\\images_for_test\\test\\" #Path to store face image captured
numbers_Of_Pic_Collect = 100 #Numbers of face image need capture

face_cascade = cv.CascadeClassifier(cascPath)

video_capture = cv.VideoCapture(0)

count = 0

while True:

    count += 1

    image_path = system_path + "image" + str(count) + ".jpg"

    ret, frame = video_capture.read(0)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        
    for (x, y, w, h) in faces:
            cv.rectangle(frame, (x,y), (x + w, y + h,), (255, 0, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            if count <= numbers_Of_Pic_Collect:
                # img = Image.fromarray(roi_gray)
                # img.save(image_path)
                img = frame[y:y+h, x:x+w]
                cv.imwrite(image_path, img)


    cv.imshow('img', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
        
video_capture.release()
cv.destroyAllWindows
# imgUrl = input("Enter image path: ")

# if len(imgUrl) > 0 :

#    img = cv.imread("C:\\Users\\Acer\\Desktop\\images_for_test\\" + imgUrl)

#     if img is None: exit()

#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     for (x, y, w, h) in faces:
#         cv.rectangle(img, (x,y), (x + w, y + h,), (255, 0, 0, 0), 2)

#     cv.imshow('img', img)
#     cv.waitKey(0)
#     cv.destroyAllWindows

# else: print("Image Path not valid")