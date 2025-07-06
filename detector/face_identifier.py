# detector/face_identifier.py
import face_recognition
import os
import cv2
import numpy as np

# Cargar imágenes del dataset y sus codificaciones
face_encodings = []
face_names = []

def cargar_rostros(dataset_path='dataset/faces'):
    for persona in os.listdir(dataset_path):
        persona_path = os.path.join(dataset_path, persona)
        if not os.path.isdir(persona_path):
            continue
        for img_name in os.listdir(persona_path):
            img_path = os.path.join(persona_path, img_name)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                face_encodings.append(encodings[0])
                face_names.append(persona)
                print(f"✅ Rostro cargado: {persona}/{img_name}")

cargar_rostros()

def identificar_rostros(frame, personas_detectadas):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encs = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encs):
        matches = face_recognition.compare_faces(face_encodings, face_encoding, tolerance=0.45)
        name = "Desconocido"

        if True in matches:
            first_match_index = matches.index(True)
            name = face_names[first_match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        print(f"[ROSTRO] {name} detectado en ({left},{top})")
