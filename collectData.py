import cv2
import time 
import os
 
#create directory
access_rights = 0o755

try:
    os.mkdir("./Dataset", access_rights)
except FileExistsError:
    pass
# Opens the Video file
cap= cv2.VideoCapture(0)
i=0

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') #load pre-trained model

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    #??? if the picture has face, it will save. Or else, It won't
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,  #an image
        scaleFactor=1.1, #Parameter specifying how much the image size is reduced at each image scale e.g reduce size by 10%
        minNeighbors=5,  #Higher value results in less detections but with higher quality. 3~6 is a good value 
        minSize=(30, 30) #Minimum possible object size. Objects smaller than that are ignored       
      )
    try:
        if faces.any():
            cv2.imwrite('Dataset/image_'+str(i)+'.jpg',frame)
            i+=1
            print("done")
    except:
        print("Error")
        pass
    time.sleep(5)
 
cap.release()
cv2.destroyAllWindows()
