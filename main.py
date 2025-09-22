import cv2              #Image/Vedio Processing Library
import numpy as np      #Numerical Computing Library

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load age and gender models
age_net = cv2.dnn.readNet("age_net.caffemodel", "age_deploy.prototxt")
gender_net = cv2.dnn.readNet("gender_net.caffemodel", "gender_deploy.prototxt")

# Mean values for preprocessing
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Glassmorphism-style overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (30, 30), (610, 450), (255, 255, 255), -1)
    frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

    for (x, y, w, h) in faces:
        # Extract face region
        face_img = frame[y:y+h, x:x+w]
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Predict gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]

        # Predict age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]

        # Draw bounding box and labels
        label = f"{gender}, {age}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Show the frame
    cv2.imshow("SmartCam", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#Steps to run the code:

#python -m venv venv

#python -m venv venv
#>>  venv\Scripts\activate

#pip install opencv-python

#python.exe -m pip install --upgrade pip

#pip install opencv-contrib-python

#python -c "import cv2; print(cv2.__version__)"

#python main.py

#Steps to create a GitHub repository and push the code:
# git init
#git add .
#git commit -m "Initial commit"
#git branch -M main
#git remote add origin https://github.com/YOUR_USERNAME/opencv-project.git
#git push -u origin main
