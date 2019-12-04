import cv2
import time
import os
import imutils
import numpy as np


name = input("Name is: ")
DIR = "Dataset/" + str(name) + "/"

cwd = os.getcwd()
pathSeparator = "/"

try:
    os.makedirs(DIR)
except FileExistsError:
    print("Folder exists")
    pass
except FileNotFoundError:
    print("can not create file")
# Opens the Video file

# i = 0

# load our serialized face detector from disk
print("[INFO] loading face detector...")
detector = cv2.dnn.readNetFromCaffe(cwd + pathSeparator + "facerecognition-master" + pathSeparator
                                    + "face_detection_model" + pathSeparator + "deploy.prototxt",
                                    cwd + pathSeparator + "facerecognition-master" + pathSeparator
                                    + "face_detection_model" + pathSeparator + "res10_300x300_ssd_iter_140000.caffemodel")


def gather(sideOfFace, name):
    # name = input("Name is: ")
    DIR = "Dataset/" + str(name) + "/" + sideOfFace + "/"
    print(DIR)
    try:
        os.makedirs(DIR)
    except FileExistsError:
        print("Folder exists")
        pass
    except FileNotFoundError:
        print("can not create file")
    cap = cv2.VideoCapture(0)
    counter = 0

    while(cap.isOpened() and counter <= 20):
        bufferImage = []
        bufferNum = 0
        while(cap.isOpened() and bufferNum < 5):
            ret, frame = cap.read()
            bufferImage.append(frame)
            bufferNum += 1
        #cv2.imshow("image", frame)
        # time.sleep(0.1)
        if ret == False:
            break
        # ??? if the picture has face, it will save. Or else, It won't
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]
        # print("shape ", frame.shape)

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()

        if len(detections) > 0:
            # we're making the assumption that each image has only ONE
            # face, so find the bounding box with the largest probability
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                # print("confidence ", confidence)
                # compute the (x, y)-coordinates of the bounding box for
                # the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                roi_color = frame[startY:endY, startX:endX]
                cv2.imwrite(DIR + "\\" + sideOfFace + "_" + str(name) + '_' +
                            str(counter)+'.jpg', roi_color)
                counter += 1
                cv2.imshow('image', roi_color)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        time.sleep(0.2)
    cap.release()
    cv2.destroyAllWindows()

    # cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('image', 600,600)
while 1:
    sideOfFace = input("enter side of face: ")
    gather(sideOfFace, name)
    if input("choose (Y/N): ").lower() == "n":
        break
