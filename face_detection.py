import cv2

def detect_faces(cascade, image, scaleFActor = 1.1):
    #Create a copy of the image
    image_copy = image.copy()
    #Covert the image to gray scale
    image_gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    #Apply the haar classifier to detect faces
    faces = cascade_face.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=5)
    #Print the no of faces found
    print('Faces found: ', len(faces))
    # Loop over coordinates and draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image_copy

#Loading the image to be tested
path = 'Resources/test.jpg'
img = cv2.imread(path)

#Converting to grayscale and resize the image
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Loading the classifier for frontal face
cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#Call the function to detect faces
faces_img = detect_faces(cascade_face, img)

cv2.imshow("Gray", img_gray)
cv2.imshow("Face detected", faces_img)
cv2.waitKey(0)

# path_alina = 'Train/Alina/1.jpg'
# path_unknown = 'Test/Alina/1.jpg'
# alina_img = face_recognition.load_image_file(path_alina)
# unkown_img = face_recognition.load_image_file(path_unknown)
#
# try:
#     alina_face_encoding = face_recognition.face_encodings(alina_img)[0]
#     unkown_face_encoding = face_recognition.face_encodings(unkown_img)[0]
# except IndexError:
#     print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
#     quit()
#
# known_faces = [
#     alina_face_encoding
# ]
#
# results = face_recognition.compare_faces(known_faces, unkown_face_encoding)