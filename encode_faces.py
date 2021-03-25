import cv2
from imutils import paths
import os
import face_recognition
import pickle

#image_paths = list(paths.list_images('Train/Alina'))
#image_paths = list(paths.list_images('dataset/lfw-funneled/'))
known_encodings = []
known_names = []

arr = os.listdir('./dataset/lfw_funneled')
print(arr)

# Loop over each folder
for i in range(len(arr)):
    # Loop over the image paths
    path = './dataset/lfw_funneled/' + arr[i]
    images = list(paths.list_images(path))

    # Get the name of the person in image
    name = arr[i]
    print(name)

    for (i, img_path) in enumerate(images):
        print("[INFO] processing image {}/{}".format(i+1, len(images)))
        image = cv2.imread(img_path)
        image = cv2.resize(image, (400, 500), interpolation = cv2.INTER_AREA)
        boxes = face_recognition.face_locations(image, model="cnn")

        # Compute the facial embedding for the face
        encodings = face_recognition.face_encodings(image, boxes)
        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(name)

print("[INFO] serializing encodings...")
data = {"encodings": known_encodings, "names": known_names}
f = open("encodings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()