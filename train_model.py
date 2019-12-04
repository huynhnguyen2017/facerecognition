# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
# import argparse
import pickle
# from sklearn.neighbors import KNeighborsClassifier


# load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open("output/embeddings.pickle", "rb").read())

# print(data['names'])
# encode the labels
# print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])
# print(labels)

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training model...")
# with SVM
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# with K nearest neighborhood
# neigh = KNeighborsClassifier(n_neighbors=5)
# neigh.fit(data["embeddings"], labels)

# print(data["embeddings"])
# write the actual face recognition model to disk
f = open("output/recognizer.pickle", "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
f = open("output/le.pickle", "wb")
f.write(pickle.dumps(le))
f.close()
