import sys
sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')

from mtcnn.mtcnn import MTCNN
from imutils import paths
import face_preprocess
import numpy as np
import argparse
import cv2
import os

ap = argparse.ArgumentParser()

ap.add_argument("--video", required=True,
                help="Number of faces that camera will get")
ap.add_argument("--output", default="../datasets/unlabeled_faces",
                help="Path to faces output")

args = vars(ap.parse_args())

# Detector = mtcnn_detector
detector = MTCNN()
# initialize video stream
cap = cv2.VideoCapture(args["video"])

# Setup some useful var
faces = 0
frames = 0


while (cap.isOpened()):
    ret, frame = cap.read()
    frames += 1
    if frames%10 == 0:
        # Get all faces on current frame
        bboxes = detector.detect_faces(frame)

        if len(bboxes) != 0:
            # Get only the biggest face
            for bboxe in bboxes:
                bbox = bboxe["box"]
                bbox = np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
                landmarks = bboxe["keypoints"]

                # convert to face_preprocess.preprocess input
                landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0], landmarks["mouth_left"][0], landmarks["mouth_right"][0],
                                     landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1], landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
                landmarks = landmarks.reshape((2,5)).T
                nimg = face_preprocess.preprocess(frame, bbox, landmarks, image_size='112,112')
                if not(os.path.exists(args["output"])):
                    os.makedirs(args["output"])
                cv2.imwrite(os.path.join(args["output"], "{}.jpg".format(faces+1)), nimg)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                print("[INFO] {} faces detected".format(faces+1))
                faces += 1
    cv2.imshow("Face detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
