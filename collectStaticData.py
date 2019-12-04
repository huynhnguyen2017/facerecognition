import cv2
import time
import os
from imutils import paths
import imutils
import numpy as np


# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join(
    ["facerecognition-master/face_detection_model", "deploy.prototxt"])
modelPath = os.path.sep.join(["facerecognition-master/face_detection_model",
                              "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


def collectStaticData(path):  # path point to folder including dataset
    if os.path.exists(path):
        imagePaths = list(paths.list_images(path))
        print(imagePaths)
        i = 0
        for image in imagePaths:
            i += 1
            # take image from file
            frame = cv2.imread(image)
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print(frame)
            # ??? if the picture has face, it will save. Or else, It won't
            frame = imutils.resize(frame, width=600)
            (h, w) = frame.shape[:2]
            # print("shape ", bufferframe[2].shape)

            # construct a blob from the frame
            frameBlob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False)

            # apply OpenCV's deep learning-based face detector to localize
            # faces in the input image
            detector.setInput(frameBlob)
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
                    # save image into old file with path (  image variable )
                    cv2.imwrite(image, roi_color)
    else:
        print("Folder not exist.")


DIR = "Dataset/"
collectStaticData(DIR)
