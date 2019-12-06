import numpy as np
import imutils
import pickle
import cv2
import os
import operator

# set path rule for both window and ubuntu
cwd = os.getcwd()
pathSeparator = "/"
if "\\" in cwd:
    pathSeparator = "\\"

# load our serialized face detector from disk
print("[INFO] loading face detector...")
detector = cv2.dnn.readNetFromCaffe(cwd + pathSeparator + "facerecognition-master" + pathSeparator
                                    + "face_detection_model" + pathSeparator + "deploy.prototxt",
                                    cwd + pathSeparator + "facerecognition-master" + pathSeparator
                                    + "face_detection_model" + pathSeparator + "res10_300x300_ssd_iter_140000.caffemodel")

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
cap = cv2.VideoCapture(0)


def ten_image_average(image):
    frame = 0
    frame = imutils.resize(image, width=600)
    (h, w) = frame.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()
    print("*******************")
    print(detections)
    if len(detections) > 0:

        # we're making the assumption that each image has only ONE
        # face, so find the bounding box with the largest probability
        # print("+++++++++++++++++++++++++")

        i = np.argmax(detections[0, 0, :, 2])
        # print(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the face
            # print(w, h, w, h)
            # print("-------------------")
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            # print(detections[0, 0, i, 3:7])
            (startX, startY, endX, endY) = box.astype("int")
            # print("+++++++++++++")
            # print(startX, startY, endX, endY)
            # extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 150 or fH < 100:
                return
            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                             (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()
            # print(vec)
            # perform classification to recognize the face
            # and return the probability of each classesq

            preds = recognizer.predict_proba(vec)[0]
            # get position of maximum value
            j = np.argmax(preds)
            proba = preds[j]
            # get  label
            name = le.classes_[j]

            if proba <= 0.6 or name == 'unknown1':
                name = 'unknown'

            data = str(name) + ' : ' + str(round(proba, 2))
            print(data)  # print label + probability
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(frame, data, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    return (name, proba, frame)  # return label, probability, image


count = 0
while True:
    # grab the frame from the threaded video stream
    # shots = []
    count += 1
    if count == 10:
        break
    ret, frame = cap.read()
    # shots.append(frame)
    # resize the frame to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image
    # dimensions

    # array contains 10 images   ???????????????????? named as frame
    # print("hello")
    try:
        (name, proba, image) = ten_image_average(frame)
        # cv2.imshow("Frame", image)
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord("q"):
        #     break
    except:
        pass

# do a bit of cleanup
cap.release()
cv2.destroyAllWindows()
# vs.stop()
