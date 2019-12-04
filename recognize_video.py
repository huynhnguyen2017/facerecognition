# Idea ==> get the rates, if the number of accuracy is more than 50%. The confirmation is accepted

# import the necessary packages
# from imutils.video import VideoStream
# from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
# import time
import cv2
import os
import operator

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# check = False
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector from disk
cwd = os.getcwd()
pathSeparator = "/"
if "\\" in cwd:
    pathSeparator = "\\"


# cascPath = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# cascEyePath = cv2.data.haarcascades + 'haarcascade_eye.xml'

# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# load our serialized face detector from disk
print("[INFO] loading face detector...")
detector = cv2.dnn.readNetFromCaffe(cwd + pathSeparator + "facerecognition-master" + pathSeparator
                                    + "face_detection_model" + pathSeparator + "deploy.prototxt",
                                    cwd + pathSeparator + "facerecognition-master" + pathSeparator
                                    + "face_detection_model" + pathSeparator + "res10_300x300_ssd_iter_140000.caffemodel")

# protoPath = os.path.sep.join(["face_detection_model", "deploy.prototxt"])
# modelPath = os.path.sep.join(["face_detection_model", "res10_300x300_ssd_iter_140000.caffemodel"])
# detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch("nn4.small2.v1.t7")

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(
    cwd + pathSeparator + "output" + pathSeparator + "recognizer.pickle", "rb").read())
le = pickle.loads(open(cwd + pathSeparator + "output" +
                       pathSeparator + "le.pickle", "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
# vs = VideoStream(src=0).start()
# time.sleep(2.0)

# # start the FPS throughput estimator
# fps = FPS().start()
lastX = 0
lastY = 0
value_array = []
# faceCascade = cv2.CascadeClassifier(
#     'haarcascade_frontalface_default.xml')
# counter = x = y = w = h = 0
cap = cv2.VideoCapture(0)
# loop over frames from the video file stream


def ten_image_average(image):
    frame = 0
    frame = imutils.resize(image, width=600)
    (h, w) = frame.shape[:2]
    # print("shape ", frame.shape)
    # print(frame)
    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()
    # print(detections.shape[1])
    # loop over the detections

    # print("here")
    if len(detections) > 0:

        # we're making the assumption that each image has only ONE
        # face, so find the bounding box with the largest probability
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            # print(frame)
            # print("confidence ", confidence)
            # compute the (x, y)-coordinates of the bounding box for
            # the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            # if fW < 150 or fH < 100:
            #     # print("here")
            #     continue
            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                             (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()
            # print(vec)
            # print(fH, fW)
            # perform classification to recognize the face
            # print(recognizer)
            preds = recognizer.predict_proba(vec)[0]
            # print(preds)
            j = np.argmax(preds)
            proba = preds[j]
            # print(le.classes_[:])
            name = le.classes_[j]
            # print(name, "\n", proba)
            # print(name)
            # if name not in stackprob.keys():
            #     stackprob[name] = {}  # create nested dicitonary
            #     # assign probability to person's dictionary
            #     stackprob[name][name] = proba
            #     # count the number of assigning
            #     stackprob[name]["counter"] = 1
            #     # add the frame to person dictionary
            #     stackprob[name]["frame"] = frame
            #     stackprob[name]["max"] = proba
            #     # print(name, " ", proba)

            # else:
            #     # check whether later proba more than previous one
            #     if proba > stackprob[name]["max"]:
            #         # this is suitable for high probability
            #         # check and overwrite the new probability
            #         stackprob[name]["max"] = proba
            #         stackprob[name]["frame"] = frame  # overwrite new frame
            #     # addition probability to variable
            #     stackprob[name][name] += proba
            #     # count the number of addition
            #     stackprob[name]["counter"] += 1

            # draw the bounding box of the face along with the
            # associated probability
            # print(startX, startY, endX, endY)

# correct
            if proba <= 0.6 or name == 'unknown1':
                name = 'unknown'

            data = str(name) + ' : ' + str(round(proba, 2))
            print(data)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(frame, data, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            # print(frame)
    # if the `q` key was pressed, break from the loop

    # r_name = None  # save image label
    # r_proba = 0         # save the return probability
    # r_image = None   # save the return image
    # try:
    #     # get maximum value from dictionay
    #     for key, value in stackprob.items():  # key variable (dictionary) is the total of probability of the particular person

    #         if value["max"] > r_proba:
    #             r_name = key
    #             r_proba = value[key] / value["counter"]
    #             r_image = value["frame"]
    #     print(r_name, " ", r_proba)
    # print(frame)
    try:
        return (name, proba, frame)  # return label, probability, image
    except:
        return


while True:
    # grab the frame from the threaded video stream
    # shots = []

    ret, frame = cap.read()
    # shots.append(frame)
    # resize the frame to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image
    # dimensions

    # array contains 10 images   ???????????????????? named as frame
    # print("hello")
    try:
        (name, proba, image) = ten_image_average(frame)
        cv2.imshow("Frame", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    except:
        pass
    # break
# stop the timer and display FPS information
# fps.stop()
# print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
# print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cap.release()
cv2.destroyAllWindows()
# vs.stop()
