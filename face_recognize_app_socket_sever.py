import socket
import threading
import numpy as np
from array import *
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

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# check = False
# ap.add_argument("-c", "--confidence", type=float, default=0.5,
#                 help="minimum probability to filter weak detections")
# args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
detector = cv2.dnn.readNetFromCaffe(cwd + pathSeparator + "facerecognition-master" + pathSeparator 
    + "face_detection_model" + pathSeparator + "deploy.prototxt", 
    cwd + pathSeparator + "facerecognition-master" + pathSeparator 
    + "face_detection_model" + pathSeparator + "res10_300x300_ssd_iter_140000.caffemodel")
# protoPath = os.path.sep.join(["face_detection_model", "deploy.prototxt"])
# modelPath = os.path.sep.join(["face_detection_model",
#                               "res10_300x300_ssd_iter_140000.caffemodel"])
# detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch("nn4.small2.v1.t7")

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(cwd + pathSeparator + "output" + pathSeparator + "recognizer.pickle", "rb").read())
le = pickle.loads(open(cwd + pathSeparator + "output" + pathSeparator + "le.pickle", "rb").read())


def ten_image_average(frames):
    stackprob = {}
    track_stack = {}
    named_frame = {}
    for frame in frames:
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
        # print(detections.shape[1])
        # loop over the detections
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

                # extract the face ROI
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 150 or fH < 100:
                    continue
                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # perform classification to recognize the face
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]
                # print(name)
                if name not in stackprob.keys():
                    stackprob[name] = proba
                    track_stack[name] = 1
                    named_frame[name] = frame
                    print(name, " ", proba)
                else:
                    stackprob[name] += proba
                    track_stack[name] += 1
                # if (proba >= 0.8) and (named_frame[name] != None):
                #     named_frame[name] = frame
                # draw the bounding box of the face along with the
                # associated probability
                text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    try:
        # get maximum value from dictionay
        maxi = max(stackprob.items(), key=operator.itemgetter(1))[0]
        # get the average of range
        max_value = stackprob[maxi] / track_stack[maxi]
        print(maxi, " ", max_value)
    except:
        return
    if(maxi == "unknow"):
        return (maxi, max_value, None)
    return (maxi, max_value, named_frame[name])


def run(frameLists):
    return ten_image_average(frameLists)

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

def createFolder(folderName):
    cwd = os.getcwd()
    pathSeparator = "/"
    if "\\" in cwd:
        pathSeparator = "\\"
    image_path = cwd + pathSeparator + "TrainImage" + pathSeparator + folderName
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    return image_path + pathSeparator

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


def collectStaticData(imageArray):  # path point to folder including dataset
    faceArray = []
    for image in imageArray:
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
                roi_color = frame[startY:endY, startX:endX]
                faceArray.append(roi_color)
                # cv2.imwrite(image, roi_color) # save image into old file with path (  image variable )

    return faceArray


def train(clientSocket, clientSocketArrdress, clientSocketPort):
    studentId = ""
    arrayOfByte = bytearray()
    while(True):
        clientSocket.sendall(b"OK\n")
        try:
            byteReceived = bytearray(clientSocket.recv(1024))
        except ConnectionResetError:
            print("Connection reset")
            return

        print("Bytes received: ", len(byteReceived))

        if(byteReceived[-1] == 10):
            byteReceived.pop(-1)
            signal = byteReceived.decode("utf-8")
            if(signal == "EXIT"):
                return
            studentId = signal
            print(studentId)

        arrayOfByte = handle_Image_Send_From_Client(clientSocket, clientSocketArrdress, clientSocketPort)
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


# def recognize_And_Response_Result():
    


def handlle_client(clientSocket, clientSocketArrdress, clientSocketPort):
    print ("Connection from : " + clientSocketArrdress + ":" + str(clientSocketPort))
    clientSignal = ""
    # clientSocket.sendall(b"OK\n")
    # f = open(system_path, "wb")
    while True:
        print("begin")
        listOfByteArray = []
        frameList = []
        arrayOfByte = bytearray()
        while True:
            try:
                byteReceived = bytearray(clientSocket.recv(1024*1024))
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
                    continue
                if(clientSignal == "RECOGNIZE"):
                    print("recognize")
                    clientSocket.sendall(b"OK\n")
                    continue
                

            if(byteReceived[-1] == 254):
                print("BREAK Receive client request process")
                byteReceived.pop(-1)
                arrayOfByte += byteReceived
                break

            arrayOfByte += byteReceived

        listOfByteArray = splitArrayOfByte(arrayOfByte)

        print("Amounts of frame received: ", len(listOfByteArray))

        # break

        frameList = convert_ByteDataArray_To_Mat(listOfByteArray)

        print("Amounts of frame after decode from String to Mat type: ", len(frameList))



        #loop to get final frame in framelist
        for frame in frameList:
            count = 1
            cv2.imshow('img', frame)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                
        #Recognize Process using harrcascades to detect face and coffe model to recognize face
        # frame = recognize(frame)
        ##################################################################

        #Test value without using recognize
        # recognizeResult = "tu"
        # name = recognizeResult
        ##################################################################

        #Recognize Process using deep learning to detect face and coffe model to recognize face
        try:
            (name, proba, frame) = run(frameList)
            recognizeResult = name
        except:
            frame = None
        ##################################################################
        

        #If frame is None. Recognize Process Failure so return Failure signal to client
        if(frame is None):
            clientSocket.sendall(b"FAILURE\n")
            continue

        print(recognizeResult)
        
        #Convert image from Mat type to base64 String to send to client 
        im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        output = BytesIO()
        im.save(output, format='JPEG')
        im_data = output.getvalue()
        image_data = base64.b64encode(im_data)


        print("Bytes send to client: ", len(image_data))

        # if not isinstance(image_data, str):
        #     # Python 3, decode from bytes to string
        #     image_data = image_data.decode()

        #Send response to client. With 'Success' signal to notification client recognize process is success
        #First response process:
        #Send 'Image' signal to notification client that next resonse is result of recognize process in Image format
        #Send result Image in Base64 String format
        #Send 'OK' signal to client to notification end of this response process
        clientSocket.sendall(b"SUCCESS\n")
        clientSocket.sendall(b"IMAGE\n")
        clientSocket.sendall(image_data)
        clientSocket.sendall(b"\n")
        clientSocket.sendall(b"OK\n")
        ##################################################################

        # print(recognizeResult)

        #Second response process:
        #Send 'Info' signal to notification client that next resonse is result of recognize process in Text format
        #Send result Text in String format
        #Send 'OK' signal to client to notification end of this response process
        clientSocket.sendall(b"INFO\n")
        # clientSocket.sendall(recognizeResult.encode())
        clientSocket.sendall(name.encode())
        clientSocket.sendall(b"\n")
        clientSocket.sendall(b"OK\n")
    clientSocket.close()
    return

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('', 5555))
s.listen(5)
while True:
    print("Listening")
    cli, (remhost, remport)= s.accept()
    # temp = bytearray(cli.recv(1024))
    # print("data receive: ", len(temp))
    # print("last byte: ", temp[len(temp) - 1])
    # temp.remove(temp[len(temp) - 1])
    # print("last byte: ", len(temp))
    t = threading.Thread(target=handlle_client, args=(cli, remhost, remport))
    t.start()

