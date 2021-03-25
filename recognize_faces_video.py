from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time
import cv2
import tkinter as tk
from tkinter import filedialog
import numpy as np

def faceRecognition(frame):
    # Detect the bounding box for each face in the image and compute facial embeddings for each one
    print("[INFO] recognizing faces...")
    boxes = face_recognition.face_locations(frame, model="cnn")
    encodings = face_recognition.face_encodings(frame, boxes)

    # Initialize the list of names for each face detected
    names = []

    # Loop over the facial embeddings
    for encoding in encodings:
        # Attempt to match each face in the input image to our known encodings
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        # Check to see if we have found a match
        if True in matches:
            # Find the indexes of all matched faces
            # Then count the total no of times each face was matched
            matchedIndex = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # Loop over the matched indexes and count each recognized face
            for i in matchedIndex:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # Find the face recognized max
            name = max(counts, key=counts.get)

        # Update the list of names
        names.append(name)

    # Loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # Draw the predicted face name on the image
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        if top - 15 > 15:
            y = top - 15
        else:
            y = top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

#Load the known faces
print("[INFO] loading encodings...")
with open("encodings.pickle", 'rb') as pickle_file:
    data = pickle.load(pickle_file)

#Hide the root window
root = tk.Tk()
root.withdraw()

#Load the video
print("[INFO] loading video...")
path = filedialog.askopenfilename()
video = cv2.VideoCapture(path)

while(True):
    #Capture frame by frame
    ret, frame = video.read()

    if(not ret):
        break

    # Show the image
    cv2.imshow('video', frame)

    faceRecognition(frame)

    # Show the image
    cv2.imshow('video', frame)

    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

video.release()
cv2.destroyAllWindows()