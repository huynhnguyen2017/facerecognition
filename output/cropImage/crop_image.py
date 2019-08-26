from PIL import Image
import cv2
import numpy as np
import os, os.path

i = 0
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #load pre-trained model 
name = input("what is your name: ")

# path joining version for other paths
DIR = 'Dataset/' + str(name) + "/"
  
file_counter = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
while i < file_counter:
  try:
    image = Image.open(DIR + str(name) +'_'+str(i)+'.jpg')
    image_data = np.asarray(image)
  
    faces = faceCascade.detectMultiScale(
        image_data,  #an image
        scaleFactor=1.1, #Parameter specifying how much the image size is reduced at each image scale e.g reduce size by 10%
        minNeighbors=5,  #Higher value results in less detections but with higher quality. 3~6 is a good value 
        minSize=(30, 30) #Minimum possible object size. Objects smaller than that are ignored       
      ) 
  
    x = y = w = h = 0
    for (x, y, w, h) in faces:
      pass
    area = (x, y, x+w, y+h)
    cropped_img = image.crop(area)
    cropped_img.save(DIR + str(name) +'_'+str(i)+'.jpg')
    i += 1
  except:
    i += 1
    try:
      os.remove(DIR + str(name) +'_'+str(i)+'.jpg')
    except:
      pass
    pass


