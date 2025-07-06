# detector/person_detector.py
from ultralytics import YOLO
import cv2

# Cargar modelo YOLOv8 preentrenado para detección de personas
# Puedes usar un modelo personalizado si lo tienes
model = YOLO('yolov8n.pt')  # Cambia por el path a tu modelo si es personalizado

# Clase de persona en COCO dataset es id 0
PERSON_CLASS_ID = 0

def detectar_personas(frame):
    personas = []
    results = model(frame, verbose=False)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls_id == PERSON_CLASS_ID and conf > 0.5:
            personas.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
            cv2.putText(frame, f'Persona {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
            print(f"[PERSONA] Detectada en ({x1},{y1}) → ({x2},{y2})")

    return personas
