from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import cvlib as cv

# Load model
model = load_model('gender_detection.model')

# open webcam
webcam = cv2.VideoCapture(0)

classes = ['man', 'women']

# Loop through frames
while webcam.isOpened():

    # read frame from webcame
    status, frame = webcam.read()

    # apply face detection
    face, confidence = cv.detect_face(frame)

    # Loop through detected face
    for idx, f in enumerate(face):

        # get corner points of face rectangle
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectange over box
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255), 2)

        # crop the detected face region
        face_crop = np.copy(frame[startY:endY,startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        #preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype('float') / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis = 0)

        # Apply gender detection on face
        conf = model.predict(face_crop)[0]  # model predict return a 2D matrix

        # Get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]

        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # Write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,0), 2)

    #  Display output
    cv2.imshow('gender detection', frame)

    # Press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()


