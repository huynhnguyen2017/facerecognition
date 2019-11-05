import numpy as np
import cv2 as cv
from PIL import Image
import time


img = cv.imread("C:\\Users\\Acer\\Documents\\NetBeansProjects\\FaceRecognizeApp\\ImageForTrain\\2.jpg", cv.IMREAD_UNCHANGED)
# cv.imwrite("C:\\Users\\Acer\\Documents\\NetBeansProjects\\FaceRecognizeApp\\ImageForTrain\\4.jpg", img)
cv.imshow('img', img)

if cv.waitKey(0) & 0xFF == ord('q'):
    print("done")
# image = Image.open("C:\\Users\\Acer\\Documents\\NetBeansProjects\\FaceRecognizeApp\\ImageForTrain\\2.jpg")
# image = Image.fromarray(image)
# image = image.convert('RGB')
# image.save("C:\\Users\\Acer\\Documents\\NetBeansProjects\\FaceRecognizeApp\\ImageForTrain\\4.jpg")

# cascPath = cv.data.haarcascades + 'haarcascade_frontalface_default.xml'
# system_path = "C:\\Users\\Acer\\Desktop\\images_for_test\\test\\" #Path to store face image captured
# numbers_Of_Pic_Collect = 100 #Numbers of face image need capture

# face_cascade = cv.CascadeClassifier(cascPath)

# video_capture = cv.VideoCapture(0)

# count = 0

# while True:

#     count += 1

#     image_path = system_path + "image" + str(count) + ".jpg"

#     ret, frame = video_capture.read(0)

#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # faces = faceCascade.detectMultiScale(
    #     gray,  #an image
    #     scaleFactor=1.1, #Parameter specifying how much the image size is reduced at each image scale e.g reduce size by 10%
    #     minNeighbors=5,  #Higher value results in less detections but with higher quality. 3~6 is a good value 
    #     minSize=(30, 30) #Minimum possible object size. Objects smaller than that are ignored       
    #     )

        
#     for (x, y, w, h) in faces:
#             cv.rectangle(frame, (x,y), (x + w, y + h,), (255, 0, 0, 0), 2)
#             roi_gray = gray[y:y+h, x:x+w]
#             if count <= numbers_Of_Pic_Collect:
#                 # img = Image.fromarray(roi_gray)
#                 # img.save(image_path)
#                 img = frame[y:y+h, x:x+w]
#                 cv.imwrite(image_path, img)


#     cv.imshow('img', frame)

#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break
        
# video_capture.release()
# cv.destroyAllWindows




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