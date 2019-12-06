# import the necessary packages
from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os

# load our serialized face detector from disk
# print("[INFO] loading face detector...")
# protoPath = os.path.sep.join(["face_detection_model", "deploy.prototxt"])
# modelPath = os.path.sep.join(["face_detection_model",
#                               "res10_300x300_ssd_iter_140000.caffemodel"])
# detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch("nn4.small2.v1.t7")

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images("Dataset"))

# initialize our lists of extracted facial embeddings and
# corresponding people names
knownEmbeddings = []
knownNames = []

# initialize the total number of faces processed
total = 0
# print(imagePaths)
# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i + 1,
                                                 imagePaths))
    name = imagePath.split(os.path.sep)[-2]

    # load the image, resize it to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image
    # dimensions
    image = cv2.imread(imagePath)

    # construct a blob for the face ROI, then pass the blob
    # through our face embedding model to obtain the 128-d
    # quantification of the face
    faceBlob = cv2.dnn.blobFromImage(image, 1.0 / 255,
                                     (96, 96), (0, 0, 0), swapRB=True, crop=False)
    embedder.setInput(faceBlob)
    vec = embedder.forward()

    # add the name of the person + corresponding face
    # embedding to their respective lists
    knownNames.append(name)
    knownEmbeddings.append(vec.flatten())

    total += 1

# for i in range(len(knownEmbeddings)):
#     print(len(knownEmbeddings[i]))
# dump the facial embeddings + names to disk
print("[INFO] serializing {} encodings...".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}
print(data)
f = open("output/embeddings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()
