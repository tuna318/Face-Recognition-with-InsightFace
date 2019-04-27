import sys
sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')

from imutils import paths
import face_preprocess
import numpy as np
import face_model
import argparse
import pickle
import cv2
import os

ap = argparse.ArgumentParser()

ap.add_argument("--dataset", default="../datasets/train",
                help="Path to training dataset")
ap.add_argument("--embeddings", default="outputs/embeddings.pickle")

# Argument of insightface
ap.add_argument('--image-size', default='112,112', help='')
ap.add_argument('--model', default='../insightface/models/model-y1-test2/model,0', help='path to load model.')
ap.add_argument('--ga-model', default='', help='path to load model.')
ap.add_argument('--gpu', default=0, type=int, help='gpu id')
ap.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
ap.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
ap.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')

args = ap.parse_args()

# Grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args.dataset))

# Initialize the faces embedder
embedding_model = face_model.FaceModel(args)

# Initialize our lists of extracted facial embeddings and corresponding people names
knownEmbeddings = []
knownNames = []

# Initialize the total number of faces processed
total = 0

# Loop over the imagePaths
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i+1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    # load the image
    image = cv2.imread(imagePath)
    # convert face to RGB color
    nimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    nimg = np.transpose(nimg, (2,0,1))
    # Get the face embedding vector
    face_embedding = embedding_model.get_feature(nimg)

    # add the name of the person + corresponding face
    # embedding to their respective list
    knownNames.append(name)
    knownEmbeddings.append(face_embedding)
    total += 1

print(total, " faces embedded")

# save to output
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open(args.embeddings, "wb")
f.write(pickle.dumps(data))
f.close()
