import socket
import threading
import numpy as np
from PIL import Image
import io
import base64
import argparse
import imutils
import pickle
import cv2
import os
from io import BytesIO
import operator

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


def ten_image_average(image, lock):
    frame = 0
    frame = imutils.resize(image, width=600)
    (h, w) = frame.shape[:2]
    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),  # resize image into 300x300; 1.0 not scale color; width and height of blob is 300x300
        (104.0, 177.0, 123.0), swapRB=False, crop=False) # mean substraction

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward() # [[[[ 0.00000000e+00  1.00000000e+00  9.99997854e-01  3.44877541e-01
                                    #    2.34312683e-01  6.38702691e-01  7.28359222e-01]
                                    # [ 0.00000000e+00  1.00000000e+00  1.19440041e-01  1.58040345e-01
                                    #    4.00908232e+00  8.23799133e-01  4.99118376e+00]]]]

    # print("here")
    if len(detections) > 0:

        # we're making the assumption that each image has only ONE
        # face, so find the bounding box with the largest probability
        i = np.argmax(detections[0, 0, :, 2]) # get the largest second element of every array
        confidence = detections[0, 0, i, 2] # get the value
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h]) # multiple width and height of origin image
            (startX, startY, endX, endY) = box.astype("int") # convert to integer

            # extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 150 or fH < 100:
                return
            
            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,     # scale color chanels of image
                                             (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()  # 128-d vector
            
            # perform classification to recognize the face
            # preds (probability of each classes): [[0.20759934 0.09564292 0.00780367 0.64482291 0.04413116]]
            preds = recognizer.predict_proba(vec)[0]
            # print(preds)
            j = np.argmax(preds) # get the position of maximum probability
            proba = preds[j]
            # print(le.classes_[:])
            name = le.classes_[j]

            # correct
            if proba <= 0.87 or name == 'unknown1':
                name = 'unknown'

            data = str(name) + ' : ' + str(round(proba, 2))
            print(data)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            # 0.45 * basic size of font; 2 is thickness of line; (startX, y) is the position of text
            cv2.putText(frame, data, (startX, y),  
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2) 
            
    return (name, proba, frame)  # return label, probability, image


def storeByteTypeImageToDisk(img, image_path):
    image = Image.open(io.BytesIO(img))
    image.save(image_path)


def storeMatTypeImageToDisk(img, image_path):
    cv2.imwrite(image_path, img)


def storeListOfMatTypeImageToDisk(imgList, imagePath):
    count = 1
    for img in imgList:
        storeMatTypeImageToDisk(img, imagePath + str(count) + ".jpg")
        count += 1


def createFolder(folderPath):
    cwd = os.getcwd()
    pathSeparator = "/"
    if "\\" in cwd:
        pathSeparator = "\\"
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
    return folderPath + pathSeparator


def removeFolder(folderPath):
    if os.path.exists(folderPath):
        os.remove(folderPath)


def handle_Image_Send_From_Client(clientSocket, clientSocketArrdress, clientSocketPort):
    arrayOfByte = bytearray()
    while True:
        try:
            byteReceived = bytearray(clientSocket.recv(1024*1024))
        except ConnectionResetError:
            cv2.destroyAllWindows()
            print("Connection reset")
            return

        if(byteReceived[-1] == 254):
            print("BREAK Receive client request process")
            byteReceived.pop(-1)
            arrayOfByte += byteReceived
            break

        arrayOfByte += byteReceived

    return arrayOfByte


def decode_String_To_Byte(arrayOfByte):
    # print("decode")
    string = arrayOfByte.decode("utf-8")
    imgdata = base64.b64decode(string)
    return imgdata


def decode_StringList_To_Byte(listOfByteArray):
    byteArray = []
    for arrayOfByte in listOfByteArray:
        imgdata = decode_String_To_Byte(arrayOfByte)
        byteArray.append(imgdata)
    return byteArray


def convert_ByteData_To_Mat(imgdata):
    file_bytes = np.fromstring(imgdata, np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return frame


def convert_ByteDataArray_To_Mat(listOfByteArray):
    frameList = []
    for arrayOfByte in listOfByteArray:
        imgdata = decode_String_To_Byte(arrayOfByte)
        # print(len(imgdata))
        frame = convert_ByteData_To_Mat(imgdata)
        frameList.append(frame)
    return frameList


def splitArrayOfByte(arrayOfByte):
    listOfByteArray = []
    while True:
        if 255 not in arrayOfByte:
            if(len(arrayOfByte) > 0):
                listOfByteArray.append(arrayOfByte)
            break
        x = arrayOfByte.index(255)
        listOfByteArray.append(arrayOfByte[:x])
        arrayOfByte = arrayOfByte[x + 1:]
    return listOfByteArray

# def faceDectectUsingHaarcascade(frames):
#     faceArray = []
#     for frame in frames:
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100,100))
#         for (x, y, w, h) in faces:
#             if(w < 100 and h < 100): continue
#             roi_gray = gray[y:y+h, x:x+w]
#             eyes = eye_cascade.detectMultiScale(roi_gray)
#             for (ex,ey,ew,eh) in eyes:
#                 cv2.rectangle(frame, (x,y), (x + w, y + h,), (255, 0, 0, 0), 2)
#                 faceArray.append(frame)
#     return faceArray


def collectStaticData(imageArray, lock):  # path point to folder including dataset
    faceArray = []
    for image in imageArray:
        lock.acquire()
        # take image from file
        # frame = cv2.imread(image)
        frame = image
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
                # roi_color = frame[startY:endY, startX:endX]
                roi_color = frame
                faceArray.append(roi_color)
                # cv2.imwrite(image, roi_color) # save image into old file with path (  image variable )
        lock.release()
    return faceArray


def readLineFromSocketStream(clientSocket, clientSocketArrdress, clientSocketPort):
    signal = ""
    while(True):
        try:
            byteReceived = bytearray(clientSocket.recv(1))
        except ConnectionResetError:
            print("Connection reset")
            return

        # print("Bytes received: ", len(byteReceived))

        if(byteReceived[0] != 10):
            signal += byteReceived.decode("utf-8")
            continue

        if(byteReceived[0] == 10):
            print("BREAK ReadLine process")
            print(signal)
            break
    return signal

# Convert image from Mat type to byte type


def convertMatFrameTypeToByteFrameType(frame):
    im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    output = BytesIO()
    im.save(output, format='JPEG')
    im_data = output.getvalue()
    return im_data

# def realTimeFaceDetect(clientSocket, clientSocketArrdress, clientSocketPort, lock):
#     arrayOfByte = bytearray()
#     clientSocket.sendall(b"OK\n")
#     while(True):
#         signal = readLineFromSocketStream(clientSocket, clientSocketArrdress, clientSocketPort)
#         if(signal == "EXIT"):
#             break
#         arrayOfByte = handle_Image_Send_From_Client(clientSocket, clientSocketArrdress, clientSocketPort)
#         listOfByteArray = splitArrayOfByte(arrayOfByte)
#         if(len(listOfByteArray) == 1):
#             imageArray = convert_ByteDataArray_To_Mat(listOfByteArray)
#             faceArray = imageArray
#             # faceArray = collectStaticData(imageArray, lock)
#             faceArray = faceDectectUsingHaarcascade(imageArray)
#             if(len(faceArray) != 1):
#                 clientSocket.sendall(b"FAILURE\n")
#                 continue
#             im_data = convertMatFrameTypeToByteFrameType(faceArray[-1])
#             image_data = base64.b64encode(im_data)
#             clientSocket.sendall(b"IMAGE\n")
#             clientSocket.sendall(image_data)
#             clientSocket.sendall(b"\n")


def realTimeFaceDetect(clientSocket, clientSocketArrdress, clientSocketPort, lock):
    clientSocket.sendall(b"OK\n")
    arrayOfByte = bytearray()
    studentId = readLineFromSocketStream(
        clientSocket, clientSocketArrdress, clientSocketPort)
    faceAngle = readLineFromSocketStream(
        clientSocket, clientSocketArrdress, clientSocketPort)
    folderPath = createFolder(
        cwd + pathSeparator + "TrainImage" + pathSeparator + studentId) + faceAngle
    folderPath = createFolder(folderPath)
    clientSocket.sendall(b"OK\n")
    count = 1
    while(True):
        signal = readLineFromSocketStream(
            clientSocket, clientSocketArrdress, clientSocketPort)
        if(signal == "EXIT"):
            break
        arrayOfByte = handle_Image_Send_From_Client(
            clientSocket, clientSocketArrdress, clientSocketPort)
        listOfByteArray = splitArrayOfByte(arrayOfByte)
        if(len(listOfByteArray) == 1):
            imageArray = convert_ByteDataArray_To_Mat(listOfByteArray)
            faceArray = imageArray
            faceArray = collectStaticData(imageArray, lock)
            # faceArray = faceDectectUsingHaarcascade(imageArray)
            if(len(faceArray) != 1):
                clientSocket.sendall(b"FAILURE\n")
                continue
            storeByteTypeImageToDisk(decode_String_To_Byte(
                listOfByteArray[0]), folderPath + str(count) + ".jpg")
            count += 1
            im_data = convertMatFrameTypeToByteFrameType(faceArray[-1])
            image_data = base64.b64encode(im_data)
            clientSocket.sendall(b"IMAGE\n")
            clientSocket.sendall(image_data)
            clientSocket.sendall(b"\n")


def deleteTrainImage(clientSocket, clientSocketArrdress, clientSocketPort):
    clientSocket.sendall(b"OK\n")
    studentId = readLineFromSocketStream(
        clientSocket, clientSocketArrdress, clientSocketPort)
    imageFolderPath = cwd + pathSeparator + "TrainImage" + pathSeparator + studentId
    print(imageFolderPath)
    print(os.path.exists(imageFolderPath))
    try:
        removeFolder(imageFolderPath)
    except:
        clientSocket.sendall(b"FAILED\n")
        return
    clientSocket.sendall(b"SUCCESS\n")


def train(clientSocket, clientSocketArrdress, clientSocketPort):
    studentId = ""
    arrayOfByte = bytearray()
    clientSocket.sendall(b"OK\n")
    signal = readLineFromSocketStream(
        clientSocket, clientSocketArrdress, clientSocketPort)
    if(signal == "EXIT"):
        return
    studentId = signal
    print(studentId)
    arrayOfByte = handle_Image_Send_From_Client(
        clientSocket, clientSocketArrdress, clientSocketPort)
    listOfByteArray = splitArrayOfByte(arrayOfByte)
    print("Amounts of frame received: ", len(listOfByteArray))
    # byteArray = decode_StringList_To_Byte(listOfByteArray)
    imageArray = convert_ByteDataArray_To_Mat(listOfByteArray)
    faceArray = collectStaticData(imageArray)
    imagePath = createFolder(studentId)
    storeListOfMatTypeImageToDisk(faceArray, imagePath)
    clientSocket.sendall(b"DONE\n")
    print("DONE")

    # clientSocket.sendall(b"TRAIN_SUCCESS\n")


def recognize_And_Response_Result(clientSocket, clientSocketArrdress, clientSocketPort, lock):
    arrayOfByte = bytearray()
    clientSocket.sendall(b"OK\n")
    arrayOfByte = handle_Image_Send_From_Client(
        clientSocket, clientSocketArrdress, clientSocketPort)
    if(len(arrayOfByte) <= 0):
        print("no frame")
        return
    listOfByteArray = splitArrayOfByte(arrayOfByte)
    print("Amounts of frame received: ", len(listOfByteArray))
    # byteArray = decode_StringList_To_Byte(listOfByteArray)
    imageList = convert_ByteDataArray_To_Mat(listOfByteArray)
    print("Amounts of frame after decode from String to Mat type: ", len(imageList))

    # loop to get final frame in framelist
    for frame in imageList:
        count = 1
        # cv2.imshow('img', frame)
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()

    # Recognize Process using harrcascades to detect face and coffe model to recognize face
    # frame = recognize(frame)
    ##################################################################
    # Test value without using recognize
    # recognizeResult = "tu"
    # name = recognizeResult
    ##################################################################

    # Recognize Process using deep learning to detect face and coffe model to recognize face
    frame = None
    try:
        (name, proba, frame) = ten_image_average(imageList[0], lock)
        print(name)
        recognizeResult = name
    except:
        print("Exception")
    ##################################################################

    # If frame is None. Recognize Process Failure so return Failure signal to client
    if(frame is None):
        clientSocket.sendall(b"FAILURE\n")
        return

    print(recognizeResult)

    # Convert image from Mat type to base64 String to send to client
    im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    output = BytesIO()
    im.save(output, format='JPEG')
    im_data = output.getvalue()
    image_data = base64.b64encode(im_data)

    print("Bytes send to client: ", len(image_data))

    # if not isinstance(image_data, str):
    #     # Python 3, decode from bytes to string
    #     image_data = image_data.decode()

    # Send response to client. With 'Success' signal to notification client recognize process is success
    # First response process:
    # Send 'Image' signal to notification client that next resonse is result of recognize process in Image format
    # Send result Image in Base64 String format
    # Send 'OK' signal to client to notification end of this response process
    clientSocket.sendall(b"SUCCESS\n")
    clientSocket.sendall(b"IMAGE\n")
    clientSocket.sendall(image_data)
    clientSocket.sendall(b"\n")
    # clientSocket.sendall(b"DONE\n")
    ##################################################################

    # print(recognizeResult)

    # Second response process:
    # Send 'Info' signal to notification client that next resonse is result of recognize process in Text format
    # Send result Text in String format
    # Send 'OK' signal to client to notification end of this response process
    clientSocket.sendall(b"INFO\n")
    # clientSocket.sendall(recognizeResult.encode())
    clientSocket.sendall(name.encode())
    clientSocket.sendall(b"\n")
    # clientSocket.sendall(b"DONE\n")


def handlle_client(clientSocket, clientSocketArrdress, clientSocketPort, lock):
    print("Connection from : " + clientSocketArrdress +
          ":" + str(clientSocketPort))
    clientSignal = ""
    # clientSocket.sendall(b"OK\n")
    # f = open(system_path, "wb")
    while True:
        print("begin")
        try:
            byteReceived = bytearray(clientSocket.recv(1024))
        except ConnectionResetError:
            cv2.destroyAllWindows()
            print("Connection reset")
            return

        print("Bytes received: ", len(byteReceived))

        if(byteReceived[-1] == 10):
            byteReceived.pop(-1)
            clientSignal = byteReceived.decode("utf-8")
            print(clientSignal)
            if(clientSignal == "TRAIN"):
                print("train")
                train(clientSocket, clientSocketArrdress, clientSocketPort)
            if(clientSignal == "RECOGNIZE"):
                print("recognize")
                recognize_And_Response_Result(
                    clientSocket, clientSocketArrdress, clientSocketPort, lock)
            if(clientSignal == "DETECT"):
                print("detect")
                realTimeFaceDetect(
                    clientSocket, clientSocketArrdress, clientSocketPort, lock)
            if(clientSignal == "DELETE"):
                print("DELETE")
                deleteTrainImage(
                    clientSocket, clientSocketArrdress, clientSocketPort)

        # break
    clientSocket.close()
    return


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('', 5555))
s.listen(5)
while True:
    print("Listening")
    cli, (remhost, remport) = s.accept()
    # temp = bytearray(cli.recv(1024))
    # print("data receive: ", len(temp))
    # print("last byte: ", temp[len(temp) - 1])
    # temp.remove(temp[len(temp) - 1])
    # print("last byte: ", len(temp))
    lock = threading.Lock()
    t = threading.Thread(target=handlle_client,
                         args=(cli, remhost, remport, lock))
    t.start()
