import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
# Load the pre-trained model
model = load_model('model.h5')

# Initialize video capture from the webcam 
cap = cv2.VideoCapture(0)

# Load Haar Cascade classifiers for face and eyes detection
face_cascade = cv2.CascadeClassifier("C:\\Users\\Public\\pythonProject5\\haarcascade_frontalface_default.xml")
eyes_cascade = cv2.CascadeClassifier("C:\\Users\\Public\\pythonProject5\\haarcascade_eye_tree_eyeglasses.xml")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)

    for (x, y, w, h) in faces:
        # Draw a rectangle around each detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Extract the region of interest (ROI) for mask prediction
        roi_color = frame[y:y + h, x:x + w]
        roi_color = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
        roi_color = cv2.resize(roi_color, (128, 128))
        roi_color = roi_color / 255.0
        roi_color = np.reshape(roi_color, (1, 128, 128, 3))

        # Predict mask
        prediction = model.predict(roi_color)
        label = np.argmax(prediction)
        label_text = "With Mask" if label == 1 else "Without Mask"

        # Put label text on the image
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with the annotations
    cv2.imshow("frame", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()