import numpy as np
from PIL import Image
import cv2
import os


cwd = os.getcwd()
pathSeparator = "/"
if "\\" in cwd:
    pathSeparator = "\\"

net = cv2.dnn.readNetFromCaffe(cwd + pathSeparator + "facerecognition-master" + pathSeparator 
    + "face_detection_model" + pathSeparator + "deploy.prototxt", 
    cwd + pathSeparator + "facerecognition-master" + pathSeparator 
    + "face_detection_model" + pathSeparator + "res10_300x300_ssd_iter_140000.caffemodel")



count = 0
system_path = "C:\\Users\\Acer\\Desktop\\images_for_test\\test\\"

numbers_Of_Pic_Collect = 50

video_capture = cv2.VideoCapture(0)

while True:

    count += 1

    ret, frame = video_capture.read(0)

    image_path = system_path + "image" + str(count) + ".jpg"

    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            text = "{:.2f}%".format(confidence * 100)

            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                (0, 0, 255), 2)

            cv2.putText(frame, text, (startX, y),
			    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            roi_gray = gray[startY:endY, startX:endX]

            # if count <= numbers_Of_Pic_Collect:
            #     img = Image.fromarray(roi_gray)
            #     # img.save(image_path)
            #     # img = frame[y:y+h, x:x+w]
            #     # cv.imwrite(image_path, img)

    cv2.imshow('img', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
video_capture.release()
cv2.destroyAllWindows