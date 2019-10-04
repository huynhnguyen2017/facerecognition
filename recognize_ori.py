'''
haar cascade version: lỗi trong dò tìm ảnh, ví dụ dù nhận diện được hay không nó cũng chụp
Nhưng có một ưu điểm là nó nhanh và ít tốn về dung lượng lưu trữ và xử lý
Thay vì cho nó vào mô hình thì chỉ cần lưu nó vào mảng để kiểm tra

1) Hệ thống sẽ dò tìm liên tục - camera luôn ở trong chế độ hoạt động
2) Nếu gương mặt được tìm thấy --> Camera sẽ chụp 10 ảnh (tương đối liên tục) --> Ảnh chụp truyền vào mô hình nhận dạng
3) Nếu nhận dạng 10 lần liên tục nếu đối tượng được nhận diện với trung bình cộng lớn hơn 85% thì được xem là đúng  --> lưu vào cơ sở dữ liệu
4) Chỉ lưu vào cơ sở dữ liệu một người một lần (cách loại bỏ: Lưu tạm người muốn lưu vào cơ sở dữ liệu nếu người tiếp theo là cùng một người thì không lưu)
'''

# import the necessary packages
import numpy as np
import imutils
import pickle
import cv2
import os
import time
import operator
# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join(["face_detection_model", "deploy.prototxt"])
modelPath = os.path.sep.join(["face_detection_model",
                              "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch("nn4.small2.v1.t7")

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
le = pickle.loads(open("output/le.pickle", "rb").read())

# Detecting face and capture an image

# Opens the Video file
cap = cv2.VideoCapture(0)
i = 0

faceCascade = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')  # load pre-trained model

value_array = [0, 0, 0, 0, 0, 0]
counter = 0

while (cap.isOpened()):
    # time.sleep(1)
    stackprob = {}
    track_stack = {}
    count = maxi = max_value = x = y = w = 0
    ret, frame = cap.read()
    # ??? if the picture has face, it will save. Or else, It won't
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,  # an image
        scaleFactor=1.1,  # Parameter specifying how much the image size is reduced at each image scale e.g reduce size by 10%
        minNeighbors=1,  # Higher value results in less detections but with higher quality. 3~6 is a good value
        # Minimum possible object size. Objects smaller than that are ignored
        minSize=(30, 30)
    )
    value_array[counter] = faces
    counter += 1
    for (x, y, w, h) in faces:
        roi_color = frame[y:y+h, x:x+w]

    if counter == 6:
        counter = 0
    # while(count < 10):

    # how to check whether face is recognized ???????????????????????????
    if len(str(value_array[0])) > 2 and len(str(value_array[1])) > 2 and len(str(value_array[2])) > 2 and len(str(value_array[3])) > 2 and len(str(value_array[4])) > 2 and len(str(value_array[5])) > 2:
        # construct a blob from the image
        while (count < 10):
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(
                gray,  # an image
                scaleFactor=1.1,  # Parameter specifying how much the image size is reduced at each image scale e.g reduce size by 10%
                minNeighbors=1,  # Higher value results in less detections but with higher quality. 3~6 is a good value
                # Minimum possible object size. Objects smaller than that are ignored
                minSize=(30, 30)
            )
            # value_array[counter] = faces

            for (x, y, w, h) in faces:
                roi_color = frame[y:y+h, x:x+w]

            faceBlob = cv2.dnn.blobFromImage(roi_color, 1.0 / 255, (96, 96),
                                             (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            # draw the bounding box of the face along with the associated
            # probability
            if(proba < 0.6):
                continue
            if name not in stackprob.keys():
                stackprob[name] = proba
                track_stack[name] = 1
            else:
                stackprob[name] += proba
                track_stack[name] += 1
            # stackprob.append(proba)

            # imgshow = frame
            count += 1
        # get maximum value from dictionay
        maxi = max(stackprob.items(), key=operator.itemgetter(1))[0]
        # get the average of range
        max_value = stackprob[maxi] / track_stack[maxi]
        if max_value > 0.90:
            print(maxi, " ", max_value)
            text = "{} : {:.2f}%".format(maxi, max_value)
            y = y - 10 if y - 10 > 10 else y + 10
            cv2.rectangle(frame, (x, y), (x+w, y+w),
                          (0, 0, 255), 2)
            cv2.putText(frame, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 2)
            # time.sleep(0.1)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

cap.release()
cv2.destroyAllWindows()
