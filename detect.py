import streamlit as st
import cv2
import numpy as np

# -------------------- Fonctions --------------------
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight, frameWidth = frameOpencvDnn.shape[:2]

    # Création du blob d'entrée pour le réseau de détection de visage
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, faceBoxes

# -------------------- Modèles --------------------
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

padding = 20

# -------------------- Interface Streamlit --------------------
st.title(" Gender & Age Prediction")
st.write("Upload an image or take a photo with your camera")

# Choix : upload image ou webcam
option = st.radio("Choose input type:", ("Upload Image", "Use Camera"))

if option == "Upload Image":
    img_file_buffer = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
elif option == "Use Camera":
    img_file_buffer = st.camera_input("Take a picture")

# Si une image est fournie
if img_file_buffer is not None:
    # Convertir l'image en array OpenCV
    bytes_data = img_file_buffer.getvalue()
    nparr = np.frombuffer(bytes_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Détection visage / âge / genre
    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if not faceBoxes:
        st.warning("No face detected!")
    else:
        for faceBox in faceBoxes:
            face = frame[max(0, faceBox[1]-padding):min(faceBox[3]+padding, frame.shape[0]-1),
                         max(0, faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            
            # Prédiction genre
            genderNet.setInput(blob)
            gender = genderList[genderNet.forward()[0].argmax()]
            
            # Prédiction âge
            ageNet.setInput(blob)
            age = ageList[ageNet.forward()[0].argmax()]
            
            # Ajouter texte sur l'image
            cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

    # Afficher l'image finale dans Streamlit
    st.image(cv2.cvtColor(resultImg, cv2.COLOR_BGR2RGB), channels="RGB")
